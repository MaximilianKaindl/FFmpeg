OBJS-$(CONFIG_DNN)                           += dnn/dnn_interface.o
OBJS-$(CONFIG_DNN)                           += dnn/dnn_io_proc.o
OBJS-$(CONFIG_DNN)                           += dnn/queue.o
OBJS-$(CONFIG_DNN)                           += dnn/safe_queue.o
OBJS-$(CONFIG_DNN)                           += dnn/dnn_backend_common.o
OBJS-$(CONFIG_DNN)                           += dnn/dnn_labels.o
OBJS-$(CONFIG_DNN)                           += dnn/dnn_tokenizer.o

DNN-OBJS-$(CONFIG_LIBTENSORFLOW)             += dnn/dnn_backend_tf.o
DNN-OBJS-$(CONFIG_LIBOPENVINO)               += dnn/dnn_backend_openvino.o
DNN-OBJS-$(CONFIG_LIBTORCH)                   += dnn/dnn_backend_torch.o
DNN-OBJS-$(CONFIG_LIBTORCH)                   += dnn/dnn_backend_torch_clip.o

OBJS-$(CONFIG_DNN)                           += $(DNN-OBJS-yes)
