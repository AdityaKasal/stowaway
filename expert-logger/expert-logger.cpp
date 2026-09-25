// expert-logger: records which experts the MoE router picks, per token, per layer.
//
// Hooks the "ffn_moe_topk-<layer>" tensor (the router's top-k expert ids) through
// llama.cpp's eval callback and writes one CSV row per (token, layer).
//
// usage:
//   expert-logger -m model.gguf --prompts prompts.tsv --out-dir data -n 256 [any llama.cpp flags]
//
// prompts.tsv: one prompt per line, "<category>\t<text>", with "\n" written literally for newlines.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "moe-stream.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// where each expert's weights live in the gguf file(s), so we can ask the OS to read them ahead of time
struct expert_map {
    struct slice {
        int    file;       // which split file
        size_t offset;     // file offset of expert 0 in this tensor
        size_t per_expert; // bytes per expert (experts are the outermost dimension, so each is contiguous)
    };
#ifdef _WIN32
    std::vector<const char *> views; // our own read-only mapping of each file; prefetching it fills the shared file cache

    // PrefetchVirtualMemory blocks while it walks the range, so a helper thread issues it and the model keeps running
    std::deque<std::vector<WIN32_MEMORY_RANGE_ENTRY>> queue;
    std::mutex              mtx;
    std::condition_variable cv;
    std::thread             worker;
    bool                    stop = false;
    size_t                  n_dropped = 0;

    void run_worker() {
        for (;;) {
            std::vector<WIN32_MEMORY_RANGE_ENTRY> job;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [&] { return stop || !queue.empty(); });
                if (stop) {
                    return;
                }
                job = std::move(queue.front());
                queue.pop_front();
            }
            PrefetchVirtualMemory(GetCurrentProcess(), job.size(), job.data(), 0);
        }
    }

    ~expert_map() {
        if (worker.joinable()) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                stop = true;
            }
            cv.notify_one();
            worker.join();
        }
    }
#else
    std::vector<int> fds;
#endif
    std::map<int, std::vector<slice>> layers; // layer -> its up/gate/down expert tensors
    size_t n_advised = 0;
    size_t bytes_advised = 0;

    // "model-00001-of-00003.gguf" -> all three paths; a single-file model -> itself
    static std::vector<std::string> split_paths(const std::string & path) {
        const size_t pos = path.rfind("-00001-of-");
        if (pos == std::string::npos) {
            return { path };
        }
        const int n = atoi(path.c_str() + pos + 10);
        std::vector<std::string> out;
        for (int i = 1; i <= n; i++) {
            char num[8];
            snprintf(num, sizeof(num), "%05d", i);
            out.push_back(path.substr(0, pos + 1) + num + path.substr(pos + 6));
        }
        return out;
    }

    bool open_file(const std::string & path) {
#ifdef _WIN32
        HANDLE f = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (f == INVALID_HANDLE_VALUE) {
            return false;
        }
        HANDLE m = CreateFileMappingA(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
        CloseHandle(f);
        if (!m) {
            return false;
        }
        const void * v = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        CloseHandle(m);
        views.push_back((const char *) v);
        return v != nullptr;
#else
        const int fd = open(path.c_str(), O_RDONLY);
        fds.push_back(fd);
        return fd >= 0;
#endif
    }

    bool load(const std::string & path) {
        int64_t n_expert = 0;
        const auto paths = split_paths(path);
        for (int fi = 0; fi < (int) paths.size(); fi++) {
            gguf_init_params gp = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
            gguf_context * g = gguf_init_from_file(paths[fi].c_str(), gp);
            if (!g) {
                return false;
            }
            if (fi == 0) { // model metadata lives in the first split
                const std::string arch = gguf_get_val_str(g, gguf_find_key(g, "general.architecture"));
                n_expert = gguf_get_val_u32(g, gguf_find_key(g, (arch + ".expert_count").c_str()));
            }
            const size_t data = gguf_get_data_offset(g);
            for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
                const std::string name = gguf_get_tensor_name(g, i);
                if (name.find("_exps.") == std::string::npos) {
                    continue;
                }
                const int layer = atoi(name.c_str() + 4); // "blk.<n>."
                layers[layer].push_back({ fi, data + gguf_get_tensor_offset(g, i), gguf_get_tensor_size(g, i) / n_expert });
            }
            gguf_free(g);
            if (!open_file(paths[fi])) {
                return false;
            }
        }
        return !layers.empty();
    }

    // non-blocking: tells the OS to start reading these experts into the file cache
    void advise(int layer, const int32_t * experts, int k) {
        auto it = layers.find(layer);
        if (it == layers.end()) {
            return;
        }
#ifdef _WIN32
        std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
#endif
        for (int j = 0; j < k; j++) {
            for (const auto & s : it->second) {
                const size_t off = s.offset + experts[j] * s.per_expert;
#ifdef _WIN32
                ranges.push_back({ (PVOID) (views[s.file] + off), s.per_expert });
#else
                radvisory ra = { (off_t) off, (int) s.per_expert };
                fcntl(fds[s.file], F_RDADVISE, &ra);
#endif
                n_advised++;
                bytes_advised += s.per_expert;
            }
        }
#ifdef _WIN32
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (!worker.joinable()) {
                worker = std::thread(&expert_map::run_worker, this);
            }
            // a few layers behind is still useful; further than that the model has already read those experts
            while (queue.size() >= 4) {
                queue.pop_front();
                n_dropped++;
            }
            queue.push_back(std::move(ranges));
        }
        cv.notify_one();
#endif
    }
};

struct logger_state {
    FILE * out      = nullptr; // experts.csv (null = don't log)
    int    prompt   = 0;       // current prompt index
    int    pos_base = 0;       // position of the first token in the current batch
    bool   is_gen   = false;   // false while processing the prompt, true while generating
    expert_map * prefetch = nullptr;
    pregate    * pg       = nullptr;
    std::vector<int32_t> row;
};

// called by the scheduler for every graph tensor: once with ask=true ("do you want this one?"),
// then again with ask=false after it has been computed.
static bool on_tensor(struct ggml_tensor * t, bool ask, void * user_data) {
    static const char prefix[]   = "ffn_moe_topk-";
    auto * st = (logger_state *) user_data;
    const bool is_topk = strncmp(t->name, prefix, sizeof(prefix) - 1) == 0;
    if (ask) {
        return is_topk;
    }
    if (!is_topk) {
        return true;
    }

    const int layer  = atoi(t->name + sizeof(prefix) - 1);
    const int k      = (int) t->ne[0]; // experts used per token
    const int n_tok  = (int) t->ne[1];

    st->row.resize(k);
    for (int i = 0; i < n_tok; i++) {
        // read row by row: the tensor can be a strided view, and may live in GPU memory
        ggml_backend_tensor_get(t, st->row.data(), i * t->nb[1], k * sizeof(int32_t));
        if (st->prefetch) {
            st->prefetch->advise(layer, st->row.data(), k);
        }
        if (st->pg && n_tok == 1) {
            st->pg->score(layer, st->row.data(), k);
        }
        if (!st->out) {
            continue;
        }
        fprintf(st->out, "%d,%d,%d,%d", st->prompt, st->is_gen ? 1 : 0, st->pos_base + i, layer);
        for (int j = 0; j < k; j++) {
            fprintf(st->out, ",%d", st->row[j]);
        }
        fputc('\n', st->out);
    }
    return true;
}

static std::string unescape(const std::string & s) {
    std::string r;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size() && s[i + 1] == 'n') {
            r += '\n';
            i++;
        } else {
            r += s[i];
        }
    }
    return r;
}

int main(int argc, char ** argv) {
    // our own flags, stripped out before llama.cpp's parser sees argv
    std::string prompts_path = "prompts.tsv";
    std::string out_dir      = "data";
    bool        prefetch     = false; // --prefetch: start reading the chosen experts as soon as the router picks them
    bool        log_experts  = true;  // --no-log: timing only
    bool        hook         = true;  // --no-hook: don't watch the router at all (baseline speed)
    double      cache_gb     = 0;     // --expert-cache-gb N: manage experts ourselves in N GB of RAM
    int         io_threads   = 16;    // --io-threads N: parallel reads for cache misses
    int         pregate_m    = 0;     // --pregate M: predict the next layer's experts and preload the top M (needs the cache)
    int         pregate_d    = 1;     // --pregate-depth D: guess D layers ahead
    std::vector<char *> rest = { argv[0] };
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompts") == 0 && i + 1 < argc) {
            prompts_path = argv[++i];
        } else if (strcmp(argv[i], "--out-dir") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(argv[i], "--prefetch") == 0) {
            prefetch = true;
        } else if (strcmp(argv[i], "--no-log") == 0) {
            log_experts = false;
        } else if (strcmp(argv[i], "--expert-cache-gb") == 0 && i + 1 < argc) {
            cache_gb = atof(argv[++i]);
        } else if (strcmp(argv[i], "--pregate") == 0 && i + 1 < argc) {
            pregate_m = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--pregate-depth") == 0 && i + 1 < argc) {
            pregate_d = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--io-threads") == 0 && i + 1 < argc) {
            io_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-hook") == 0) {
            hook = false;
            log_experts = false;
        } else {
            rest.push_back(argv[i]);
        }
    }

    common_params params;
    common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    std::vector<std::pair<std::string, std::string>> prompts;
    {
        std::ifstream f(prompts_path);
        if (!f) {
            fprintf(stderr, "cannot open %s\n", prompts_path.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(f, line)) {
            const size_t tab = line.find('\t');
            if (line.empty() || line[0] == '#' || tab == std::string::npos) {
                continue;
            }
            prompts.emplace_back(line.substr(0, tab), unescape(line.substr(tab + 1)));
        }
    }

    logger_state st;
    expert_map   emap;
    expert_cache ecache;
    if (cache_gb > 0 && !ecache.init(params.model.path, cache_gb, io_threads)) {
        fprintf(stderr, "could not set up the expert cache\n");
        return 1;
    }
    pregate pg;
    if (pregate_m > 0) {
        if (cache_gb <= 0 || !pg.load(params.model.path)) {
            fprintf(stderr, "--pregate needs --expert-cache-gb and F32 router weights\n");
            return 1;
        }
        pg.m     = pregate_m;
        pg.depth = pregate_d;
        pg.cache = &ecache;
        st.pg    = &pg;
        ecache.predictor = [&pg](int layer, const float * x, int n, int n_tok) { pg.predict(layer, x, n, n_tok); };
        fprintf(stderr, "pre-gating: preloading the top %d predicted experts of the next layer (%zu router tables)\n",
                pregate_m, pg.router.size());
    }
    if (prefetch) {
        if (!emap.load(params.model.path)) {
            fprintf(stderr, "could not read expert offsets from %s\n", params.model.path.c_str());
            return 1;
        }
        st.prefetch = &emap;
        fprintf(stderr, "prefetch: on, %zu MoE layers\n", emap.layers.size());
    }
    st.out = log_experts ? fopen((out_dir + "/experts.csv").c_str(), "w") : nullptr;
    FILE * f_tok    = fopen((out_dir + "/tokens.csv").c_str(), "w");
    FILE * f_prompt = fopen((out_dir + "/prompts.csv").c_str(), "w");
    if ((log_experts && !st.out) || !f_tok || !f_prompt) {
        fprintf(stderr, "cannot write to %s (does it exist?)\n", out_dir.c_str());
        return 1;
    }

    if (hook) {
        params.cb_eval           = on_tensor;
        params.cb_eval_user_data = &st;
    }
    params.warmup            = false; // a warmup run would log junk routing
    params.n_ubatch          = params.n_batch; // one graph per batch, so positions stay in order

    auto init = common_init_from_params(params);
    llama_model   * model = init->model();
    llama_context * ctx   = init->context();
    if (!model || !ctx) {
        fprintf(stderr, "failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    common_sampler * smpl = init->sampler(0);

    // header: k expert columns (k lives in the gguf metadata as "<arch>.expert_used_count")
    char arch[64] = {};
    char val[32]  = {};
    llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
    if (llama_model_meta_val_str(model, (std::string(arch) + ".expert_used_count").c_str(), val, sizeof(val)) < 0) {
        fprintf(stderr, "%s is not a MoE model (no expert_used_count)\n", arch);
        return 1;
    }
    const int k = atoi(val);
    if (st.out) {
        fprintf(st.out, "prompt,gen,pos,layer");
        for (int j = 0; j < k; j++) {
            fprintf(st.out, ",e%d", j);
        }
        fputc('\n', st.out);
    }
    fprintf(f_tok, "prompt,gen,pos,token_id,ms\n");
    fprintf(f_prompt, "prompt,category,n_prompt_tokens,n_gen_tokens,prefill_s,gen_s\n");

    for (size_t p = 0; p < prompts.size(); p++) {
        const auto & [category, text] = prompts[p];
        const std::string chat = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n";

        llama_memory_clear(llama_get_memory(ctx), true);
        common_sampler_reset(smpl);

        std::vector<llama_token> toks = common_tokenize(ctx, chat, true, true);
        if ((int) toks.size() > params.n_batch) {
            fprintf(stderr, "prompt %zu is longer than the batch size, skipping\n", p);
            continue;
        }

        st.prompt   = (int) p;
        st.pos_base = 0;
        st.is_gen   = false;

        auto t0 = std::chrono::steady_clock::now();
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int) toks.size()))) {
            fprintf(stderr, "decode failed on prompt %zu\n", p);
            return 1;
        }
        for (size_t i = 0; i < toks.size(); i++) {
            fprintf(f_tok, "%zu,0,%zu,%d,0\n", p, i, toks[i]);
        }
        auto t1 = std::chrono::steady_clock::now();

        int pos   = (int) toks.size();
        int n_gen = 0;
        st.is_gen = true;
        for (; n_gen < params.n_predict; n_gen++) {
            llama_token tok = common_sampler_sample(smpl, ctx, -1);
            common_sampler_accept(smpl, tok, true);
            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }
            auto ts = std::chrono::steady_clock::now();
            st.pos_base = pos;
            if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
                fprintf(stderr, "decode failed on prompt %zu, token %d\n", p, n_gen);
                return 1;
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ts).count();
            fprintf(f_tok, "%zu,1,%d,%d,%.2f\n", p, pos, tok, ms);
            pos++;
        }
        auto t2 = std::chrono::steady_clock::now();

        const double prefill_s = std::chrono::duration<double>(t1 - t0).count();
        const double gen_s     = std::chrono::duration<double>(t2 - t1).count();
        fprintf(f_prompt, "%zu,%s,%zu,%d,%.2f,%.2f\n", p, category.c_str(), toks.size(), n_gen, prefill_s, gen_s);
        if (st.out) {
            fflush(st.out);
        }
        fflush(f_tok);
        fflush(f_prompt);
        fprintf(stderr, "[%zu/%zu] %-10s prompt %4zu tok, generated %4d tok at %.1f tok/s\n",
                p + 1, prompts.size(), category.c_str(), toks.size(), n_gen, n_gen / gen_s);
    }

    if (st.out) {
        fclose(st.out);
    }
    if (cache_gb > 0) {
        ecache.print_stats();
    }
    if (pregate_m > 0 && hook) {
        fprintf(stderr, "pre-gating: %.1f%% of the experts each layer used had been predicted a layer ahead (%zu of %zu)\n",
                pg.total ? 100.0 * pg.right / pg.total : 0.0, pg.right, pg.total);
    }
    if (prefetch) {
        fprintf(stderr, "prefetch: %zu reads advised, %.1f GB\n", emap.n_advised, emap.bytes_advised / 1e9);
#ifdef _WIN32
        fprintf(stderr, "prefetch: %zu layer requests dropped because the helper thread fell behind\n", emap.n_dropped);
#endif
    }
    fclose(f_tok);
    fclose(f_prompt);
    return 0;
}
