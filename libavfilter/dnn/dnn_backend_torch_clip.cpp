#include "dnn_backend_torch_clip.h"
#include "dnn_backend_common.h"
#include "dnn_backend_torch_common.h"

extern "C" {
#include "libavutil/mem.h"
#include "libavutil/avassert.h"
#include "libavutil/log.h"
}

torch::Tensor get_tokens(THModel *th_model, std::string prompt) {
    DnnContext *ctx = th_model->ctx;
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

std::string LoadBytesFromFile(const std::string& path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}

int create_tokenizer(THModel *th_model, std::string tokenizer_path) {
    try {
        std::string tokenizer_path_str(tokenizer_path);
        auto blob = LoadBytesFromFile(tokenizer_path_str);
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
int encode_image_clip(THModel *th_model, THRequestItem *request) {
    THInferRequest *infer_request = request->infer_request;
    DnnContext *ctx = th_model->ctx;
    
    try {
        // Get model's device
        c10::Device device = (*th_model->jit_model->parameters().begin()).device();
        
         // First normalize to [0,1] range
        torch::Tensor image_tensor = infer_request->input_tensor->to(torch::kFloat32).div(255.0);
        
        // Apply CLIP specific normalization
        std::vector<double> mean = {0.48145466, 0.4578275, 0.40821073};
        std::vector<double> std = {0.26862954, 0.26130258, 0.27577711};
        
        for (int c = 0; c < 3; c++) {
            image_tensor.select(1, c).sub_(mean[c]).div_(std[c]);
        }
        
        // Move to correct device and get image features
        image_tensor = image_tensor.to(device);
        
        // Get image features
        auto image_features = th_model->jit_model->run_method(
            "encode_image",
            image_tensor,
            true  // normalize
        );

        if (!image_features.isTensor()) {
            av_log(ctx, AV_LOG_ERROR, "Model returned invalid non-tensor output\n");
            return AVERROR(EINVAL);
        }

        torch::Tensor encoded = image_features.toTensor();

        *infer_request->input_tensor = encoded;
        return 0;

    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Image encoding error: %s\n", e.what());
        return AVERROR(EINVAL);
    }
}

int encode_text_clip(THModel *th_model, THRequestItem *request) {
    THInferRequest *infer_request = request->infer_request;
    DnnContext *ctx = th_model->ctx;
    THClipContext *clip_ctx = th_model->clip_ctx;
    infer_request->text_embeddings = new std::vector<torch::Tensor>();

    int ret = create_tokenizer(th_model,th_model->clip_ctx->tokenizer_path);
    if(ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Error creating tokenizer\n");
        return ret;
    }

    try {
        infer_request->text_embeddings->reserve(clip_ctx->labels.size());

        c10::Device device = (*th_model->jit_model->parameters().begin()).device();

        for (const auto& label : clip_ctx->labels) {
            torch::Tensor tokens = get_tokens(th_model, label);
            tokens = tokens.to(device);

            auto encoded_result = th_model->jit_model->run_method("encode_text", tokens, true);
            
            if (!encoded_result.isTensor()) {
                av_log(ctx, AV_LOG_ERROR, "Model returned invalid non-tensor output for text encoding\n");
                return AVERROR(EINVAL);
            }
            torch::Tensor encoded_tensor = encoded_result.toTensor();
            infer_request->text_embeddings->push_back(encoded_tensor.clone());
        }
        
        return 0;
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Text encoding error: %s\n", e.what());
        return AVERROR(EINVAL);
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
    int ret;

    *infer_request->input_tensor = torch::from_blob(input.data,
    {1, 3, 224, 224},
    deleter, 
    torch::kFloat32);
    
    *infer_request->output = infer_request->input_tensor->clone().detach();
    // Verify the clone worked
    if (!infer_request->output->defined() || infer_request->output->sizes() != infer_request->input_tensor->sizes()) {
        av_log(ctx, AV_LOG_ERROR, "Tensor cloning failed\n");
        return AVERROR(EINVAL);
    }

    ret = encode_image_clip(th_model, request);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Image encoding failed in CLIP preprocessing\n");
        return ret;
    }

    ret = encode_text_clip(th_model, request);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Text encoding failed in CLIP preprocessing\n");
        return ret;
    }

    return 0;
}
void softmax(std::vector<std::pair<float, std::string>>& labels) {
    // Find max for numerical stability
    float max_score = -std::numeric_limits<float>::infinity();
    for (const auto& label : labels) {
        max_score = std::max(max_score, label.first);
    }
    
    // Compute exp(x-max) and sum
    float sum_exp = 0.0f;
    for (auto& label : labels) {
        label.first = std::exp(label.first - max_score);
        sum_exp += label.first;
    }
    
    // Normalize
    for (auto& label : labels) {
        label.first = (label.first / sum_exp);
    }
}


void print_clip_similarity_scores(THModel *th_model, const std::vector<std::pair<float, std::string>>& scored_labels, DnnContext *ctx) {
    try {
        av_log(ctx, AV_LOG_INFO, "\nCLIP Analysis Results:\n");
        // Create a mutable copy for sorting
        std::vector<std::pair<float, std::string>> sorted_labels = scored_labels;
        softmax(sorted_labels);
        std::sort(sorted_labels.begin(), sorted_labels.end(),
                 std::greater<std::pair<float, std::string>>());
        
        av_log(ctx, AV_LOG_INFO, "\nRanked Matches:\n");
        // Remove const from the loop variable
        for (auto& scored_label : sorted_labels) {
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
        
        if (!sorted_labels.empty()) {
            av_log(ctx, AV_LOG_INFO, "\nBest match: \"%s\" with %.1f%% confidence\n",
                   sorted_labels[0].second.c_str(),
                   sorted_labels[0].first);
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

torch::Tensor process_clip_similarity(const torch::Tensor& image_features, 
                                    const torch::Tensor& text_embedding,
                                    DnnContext *ctx,
                                    float temperature = 0.07) {                                  
    auto image_f = torch::nn::functional::normalize(image_features, 
        torch::nn::functional::NormalizeFuncOptions().dim(-1));
    auto text_f = torch::nn::functional::normalize(text_embedding,
        torch::nn::functional::NormalizeFuncOptions().dim(-1));
    
    try{
        auto similarity = torch::matmul(image_f, text_f.transpose(0, 1));
        return similarity.div(temperature);
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Matrix multiplication failed. Shapes: image %ldx%ld, text %ldx%ld\n",
            image_features.size(0), image_features.size(1),
            text_embedding.size(0), text_embedding.size(1));
        throw;
    }
}

int process_clip_inference(THModel *th_model, THInferRequest *infer_request, 
                                const c10::Device& device, DnnContext *ctx) {
    try {
        std::vector<std::pair<float, std::string>> scored_labels;
        scored_labels.reserve(th_model->clip_ctx->labels.size());
        int i = 0;
        torch::Tensor image_features = infer_request->input_tensor->to(device);

        for(auto &text_embedding : *(infer_request->text_embeddings)) {
            auto text_embedding_device = text_embedding.to(device);
            auto similarity = process_clip_similarity(image_features, text_embedding_device, ctx);
            float sim_value = (similarity).item<float>();

            scored_labels.push_back({sim_value, th_model->clip_ctx->labels[i]});
            i++;
        }
        print_clip_similarity_scores(th_model, scored_labels, ctx);              
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Inference error: %s\n", e.what());
        return AVERROR(EINVAL);
    }
    return 0;
}