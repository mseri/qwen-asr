# qwen_asr — Qwen3-ASR Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files
SRCS = qwen_asr.c qwen_asr_kernels.c qwen_asr_kernels_generic.c qwen_asr_kernels_neon.c qwen_asr_kernels_avx.c qwen_asr_audio.c qwen_asr_encoder.c qwen_asr_decoder.c qwen_asr_tokenizer.c qwen_asr_safetensors.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = qwen_asr
EXTRA_OBJS =

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug info help blas mps test bench test-stream-cache

# Default: show available targets
all: help

help:
	@echo "qwen_asr — Qwen3-ASR Pure C Inference - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make blas     - With BLAS acceleration (Accelerate/OpenBLAS)"
	@echo "  make mps      - Apple Silicon (fastest, macOS only)"
	@echo ""
	@echo "Other targets:"
	@echo "  make debug    - Debug build with AddressSanitizer"
	@echo "  make test     - Run regression suite + quick benchmark (requires ./qwen_asr and model files)"
	@echo "  make bench    - Run benchmark only (3 runs, all modes)"
	@echo "  make test-stream-cache - Run stream cache on/off equivalence check"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo ""
	@echo "Example: make mps && ./qwen_asr -d model_dir -i audio.wav"

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
blas: LDFLAGS += -framework Accelerate
else
blas: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
blas: LDFLAGS += -lopenblas
endif
blas:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"
	@echo ""
	@echo "Built with BLAS backend"

# =============================================================================
# Backend: mps (Metal Performance Shaders — Apple Silicon only)
# =============================================================================
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
ifneq ($(UNAME_M),arm64)
mps:
	@echo "Error: 'make mps' requires Apple Silicon (arm64). This machine is $(UNAME_M)." >&2
	@echo "Use 'make blas' instead (Apple Accelerate with AVX2)." >&2
	@exit 1
else
mps: CC = clang
mps: CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_MPS -DACCELERATE_NEW_LAPACK
mps: LDFLAGS += -framework Accelerate -framework Metal -framework MetalPerformanceShaders -framework Foundation -lobjc
mps: EXTRA_OBJS = qwen_asr_kernels_metal.o
mps:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CC=clang CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)" EXTRA_OBJS="$(EXTRA_OBJS)"
	@echo ""
	@echo "Built with Metal/MPS backend (Apple Silicon)"
endif
else
mps:
	@echo "Error: 'make mps' is only supported on macOS with Apple Silicon" >&2
	@exit 1
endif

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o $(EXTRA_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c qwen_asr.h qwen_asr_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Objective-C compilation rule (for Metal backend)
%.o: %.m qwen_asr_kernels_metal.h
	clang $(CFLAGS) -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
debug:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Utilities
# =============================================================================
clean:
	rm -f $(OBJS) main.o qwen_asr_kernels_metal.o $(TARGET)

info:
	@echo "Platform: $(UNAME_S)"
	@echo "Compiler: $(CC)"
	@echo ""
ifeq ($(UNAME_S),Darwin)
	@echo "Backends available: blas (Apple Accelerate), mps (Metal/MPS)"
else
	@echo "Backend: blas (OpenBLAS)"
endif

test:
	./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-0.6b --bench

bench:
	./asr_regression.py --binary ./qwen_asr --model-dir qwen3-asr-0.6b \
		--bench-only --bench-runs 3 --bench-modes offline,segmented,stream

# =============================================================================
# Dependencies
# =============================================================================
qwen_asr.o: qwen_asr.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h qwen_asr_audio.h qwen_asr_tokenizer.h
qwen_asr_kernels.o: qwen_asr_kernels.c qwen_asr_kernels.h qwen_asr_kernels_impl.h
qwen_asr_kernels_generic.o: qwen_asr_kernels_generic.c qwen_asr_kernels_impl.h
qwen_asr_kernels_neon.o: qwen_asr_kernels_neon.c qwen_asr_kernels_impl.h
qwen_asr_kernels_avx.o: qwen_asr_kernels_avx.c qwen_asr_kernels_impl.h
qwen_asr_audio.o: qwen_asr_audio.c qwen_asr_audio.h
qwen_asr_encoder.o: qwen_asr_encoder.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_decoder.o: qwen_asr_decoder.c qwen_asr.h qwen_asr_kernels.h qwen_asr_safetensors.h
qwen_asr_tokenizer.o: qwen_asr_tokenizer.c qwen_asr_tokenizer.h
qwen_asr_safetensors.o: qwen_asr_safetensors.c qwen_asr_safetensors.h
qwen_asr_kernels_metal.o: qwen_asr_kernels_metal.m qwen_asr_kernels_metal.h
main.o: main.c qwen_asr.h qwen_asr_kernels.h
