#include "dnn_backend_torch_clip.h"
#include "dnn_backend_common.h"

extern "C" {
#include "libavutil/mem.h"
#include "libavutil/avassert.h"
#include "libavutil/log.h"
}

torch::Tensor get_tokens(THModel *th_model, std::string prompt) {
    DnnContext *ctx = th_model->ctx;
    //TODO : Load Special Tokens from tokenizer
    const int expected_length = 77;  // CLIP's standard sequence length
    const int start_token = 49406;   // CLIP's BOS token
    const int end_token = 49407;     // CLIP's EOS token
    const int pad_token = 0;         // CLIP's padding token

    try {
        if (!th_model->clip_ctx || !th_model->clip_ctx->tokenizer) {
            throw std::runtime_error("Tokenizer not initialized");
        }

        // Create vector with correct size, filled with padding tokens
        std::vector<int64_t> padded_ids(expected_length, pad_token);
        
        // Add start token
        padded_ids[0] = start_token;
        
        try {
            // Get tokens from the tokenizer
            std::vector<int> tokens = th_model->clip_ctx->tokenizer->Encode(prompt);

            // Calculate how many tokens we can copy (leaving space for start and end tokens)
            const size_t max_text_tokens = expected_length - 2;
            
            const size_t num_tokens = tokens.size();
            if(num_tokens > max_text_tokens) {
                av_log(ctx, AV_LOG_WARNING, "Input text is too long, truncating to %ld tokens\n", max_text_tokens);
            }
            // Copy tokens after the start token
            size_t i;
            for (i = 0; i < num_tokens; i++) {
                padded_ids[i + 1] = tokens[i];
            }
            padded_ids[i+1] = end_token;

            auto tensor = torch::from_blob(
                padded_ids.data(),
                {1, expected_length}, 
                torch::kInt64
            ).clone(); 

            return tensor;

        } catch (const std::exception& e) {
            av_log(ctx, AV_LOG_ERROR, "Token encoding failed: %s\n", e.what());
            // Return empty tensor with correct dimensions on error
            return torch::zeros({1, expected_length}, torch::kInt64);
        }

    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Token generation failed: %s\n", e.what());
        return torch::zeros({1, expected_length}, torch::kInt64);
    }
}

int load_bytes_from_file(const std::string& path, std::string& data, void* log_ctx) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        av_log(log_ctx, AV_LOG_ERROR, "Cannot open file: %s\n", path.c_str());
        return AVERROR(ENOENT);
    }

    try {
        fs.seekg(0, std::ios::end);
        size_t size = static_cast<size_t>(fs.tellg());
        fs.seekg(0, std::ios::beg);

        if (fs.fail()) {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to determine file size: %s\n", path.c_str());
            return AVERROR(EIO);
        }

        data.resize(size);
        fs.read(data.data(), size);

        if (fs.fail()) {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to read file: %s\n", path.c_str());
            return AVERROR(EIO);
        }

        return 0;
    } catch (const std::exception& e) {
        av_log(log_ctx, AV_LOG_ERROR, "Exception while reading file %s: %s\n", 
               path.c_str(), e.what());
        return AVERROR(ENOMEM);
    }
}

int create_tokenizer(THModel *th_model, std::string tokenizer_path) {
    std::string blob;
    int ret = load_bytes_from_file(tokenizer_path, blob, th_model->ctx);
    if (ret < 0) {
        return ret;
    }

    try {
        th_model->clip_ctx->tokenizer = Tokenizer::FromBlobJSON(blob);
        return 0;
    } catch (const c10::Error& e) {
        av_log(th_model->ctx, AV_LOG_ERROR, "Error creating tokenizer: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

int init_clip_model(THModel *th_model, AVFilterContext *filter_ctx) {
    try {
        th_model->jit_model->get_method("encode_image");
        th_model->jit_model->get_method("encode_text");
        th_model->is_clip_model = true;
        th_model->clip_ctx = (THClipContext *)av_mallocz(sizeof(THClipContext));
        av_log(th_model->ctx, AV_LOG_INFO, 
               "Successfully initialized CLIP model\n");
        return 0;

    } catch (const c10::Error& e) {
        av_log(th_model->ctx, AV_LOG_ERROR, 
               "Error during CLIP model initialization: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

torch::Tensor get_clip_tokens_tensor(THModel *th_model, THRequestItem *request) {
    DnnContext *ctx = th_model->ctx;
    THClipContext *clip_ctx = th_model->clip_ctx;

    if (clip_ctx->labels.empty()) {
        av_log(ctx, AV_LOG_ERROR, "No labels provided for text encoding\n");
        return torch::Tensor();
    }

    try {
        std::vector<torch::Tensor> text_tokens;
        text_tokens.reserve(clip_ctx->labels.size());
        
        for (const auto& label : clip_ctx->labels) {
            auto tokens = get_tokens(th_model, label);
            if (!tokens.defined()) {
                av_log(ctx, AV_LOG_ERROR, "Failed to tokenize text: %s\n", label.c_str());
                return torch::Tensor();
            }
            text_tokens.push_back(tokens);
        }
        return torch::cat(text_tokens, 0);
    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Error preparing CLIP inputs: %s\n", e.what());
        return torch::Tensor();
    }
}

static void deleter(void *arg)
{
    av_freep(&arg);
}

int fill_model_input_clip(THModel *th_model, THRequestItem *request, DNNData input) 
{
    DnnContext *ctx = th_model->ctx;
    THInferRequest *infer_request = request->infer_request;
    *infer_request->input_tensor = torch::from_blob(
        input.data,
        {1, 3, 224, 224},
        deleter, 
        torch::kFloat32
    );

    *infer_request->output = infer_request->input_tensor->clone().detach();
    // Verify the clone worked
    if (!infer_request->output->defined() || infer_request->output->sizes() != infer_request->input_tensor->sizes()) {
        av_log(ctx, AV_LOG_ERROR, "Tensor cloning failed\n");
        return AVERROR(EINVAL);
    }

    int ret;
    ret = create_tokenizer(th_model, th_model->clip_ctx->tokenizer_path);
    if(ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Error creating tokenizer\n");
        return ret;
    }
    return 0;
}

std::vector<std::pair<float, std::string>> softmax_and_map_to_labels(const std::vector<float>& scores, const std::vector<std::string>& labels) {
    torch::Tensor scores_tensor = torch::tensor(scores);
    torch::Tensor softmax_scores = torch::softmax(scores_tensor, 0);
    
    std::vector<std::pair<float, std::string>> scored_labels;
    scored_labels.reserve(scores.size());
    
    // Access softmax scores
    auto scores_accessor = softmax_scores.accessor<float,1>();
    
    for (size_t i = 0; i < scores.size(); i++) {
        scored_labels.emplace_back(
            scores_accessor[i] * 100.0f,  // Convert to percentage
            labels[i]
        );
    }
    
    std::sort(scored_labels.begin(), scored_labels.end(),
              std::greater<std::pair<float, std::string>>());
    
    return scored_labels;
}


void print_clip_similarity_scores(THModel *th_model, const std::vector<std::pair<float, std::string>>& scored_labels, DnnContext *ctx) {
    try {
        av_log(ctx, AV_LOG_INFO, "\nCLIP Analysis Results:\n");
        for (auto& scored_label : scored_labels) {
            const float high_confidence_threshold = 50.0f;
            if (scored_label.first >= high_confidence_threshold) {
                av_log(ctx, AV_LOG_INFO, "✓ ");
            } else {
                av_log(ctx, AV_LOG_INFO, "  ");
            }
            
            av_log(ctx, AV_LOG_INFO, "%.1f%% : \"%s\"\n",
                   scored_label.first,
                   scored_label.second.c_str());
        }
        
        if (!scored_labels.empty()) {
            av_log(ctx, AV_LOG_INFO, "\nBest match: \"%s\" with %.1f%% confidence\n",
                   scored_labels[0].second.c_str(),
                   scored_labels[0].first);
        }
        
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Error processing similarity scores: %s\n", e.what());
    }
}
int set_params_clip(THModel *th_model, const char **labels, int label_count, const char *tokenizer_path) {
    if (!th_model || !labels || label_count <= 0) {
        return AVERROR(EINVAL);
    }
    std::vector<std::string> label_vector;
    label_vector.reserve(label_count); 
    
    for (int i = 0; i < label_count; i++) {
        if (labels[i]) {
            label_vector.push_back(std::string(labels[i]));
        }
    }
    th_model->clip_ctx->labels = label_vector;
    th_model->clip_ctx->tokenizer_path = tokenizer_path;
    return 0;
}

torch::Tensor normalize_features(const torch::Tensor& features, int64_t dim = 1) {
    return torch::nn::functional::normalize(features, 
        torch::nn::functional::NormalizeFuncOptions().dim(dim));
}

torch::Tensor calculate_clip_similarity_matrix(const torch::Tensor& image_features, const torch::Tensor& text_embedding, float logit_scale, DnnContext *ctx, float temperature = 1.0) {        
    auto image_f = normalize_features(image_features);
    auto text_f = normalize_features(text_embedding);

    try {
        auto similarity = torch::matmul(image_f, text_f.transpose(0,1));    
        similarity = similarity * logit_scale;      
        return similarity.div(temperature);
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Similarity computation failed: %s\n", e.what());
        return torch::Tensor();
    }
}


int extract_clip_outputs(THModel *th_model, THRequestItem *request, const c10::ivalue::Tuple* output) {
    DnnContext *ctx = th_model->ctx;
    THInferRequest *infer_request = request->infer_request;
    try {
        *infer_request->output = output->elements()[0].toTensor();

        auto text_features = output->elements()[1].toTensor();
        std::vector<torch::Tensor> text_features_split = text_features.split(1, 0);
        infer_request->text_embeddings = new std::vector<torch::Tensor>();
        infer_request->text_embeddings->reserve(text_features_split.size());
        for (const auto& text_feature : text_features_split) {
            infer_request->text_embeddings->push_back(text_feature);
        }

        torch::Tensor logit_tensor = output->elements()[2].toTensor();
        auto logit_scale = logit_tensor.item<float>();
        th_model->clip_ctx->logit_scale = std::clamp(logit_scale, 1.0f, 100.0f);

        return 0;
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Torch error extracting CLIP outputs: %s\n", e.what());
        return AVERROR(EINVAL);
    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Error extracting CLIP outputs: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

int process_clip_similarity(THModel *th_model, THRequestItem *request, c10::Device device) {
    DnnContext *ctx = th_model->ctx;
    THInferRequest *infer_request = request->infer_request;
    auto image_features = infer_request->output;
    auto text_embeddings = infer_request->text_embeddings;

    std::vector<float> similarity_scores;
    similarity_scores.reserve(text_embeddings->size());

    try {
        auto device_image = image_features->to(device);

        for (size_t i = 0; i < text_embeddings->size(); i++) {
            auto device_text_embedding = (*infer_request->text_embeddings)[i].to(device);
            auto similarity = calculate_clip_similarity_matrix(device_image, device_text_embedding, th_model->clip_ctx->logit_scale, ctx);
            float similarity_value = similarity.item<float>();
            similarity_scores.push_back(similarity_value);

            av_log(ctx, AV_LOG_DEBUG, "Label %s: logit_value=%.4f\n", 
                   th_model->clip_ctx->labels[i].c_str(), similarity_value);
        }

        auto scored_labels = softmax_and_map_to_labels(similarity_scores, th_model->clip_ctx->labels);
        print_clip_similarity_scores(th_model, scored_labels, ctx);
        return 0;

    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "CLIP similarity computation error: %s\n", e.what());
        return AVERROR(EINVAL);
    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Error computing similarities: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

void free_clip_context(THClipContext *clip_ctx) {
    if (!clip_ctx)
        return;
        
    clip_ctx->tokenizer.reset();
    clip_ctx->labels.clear();
    clip_ctx->tokenizer_path.clear();
}