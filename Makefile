CC = gcc
CFLAGS = -Wall -O2 -Iinclude -D_POSIX_C_SOURCE=200809L
LDFLAGS = -lm

# Check if OpenBLAS is available
OPENBLAS_EXISTS := $(shell pkg-config --exists openblas && echo yes)

ifeq ($(OPENBLAS_EXISTS),yes)
    CFLAGS += $(shell pkg-config --cflags openblas) -DUSE_OPENBLAS
    LDFLAGS += $(shell pkg-config --libs openblas)
else
    # Fallback if pkg-config doesn't find it but it might be there
    # Or just skip it. For now, let's try to add it if tensor_tests needs it.
    LDFLAGS += -lopenblas
endif

# Source files and objects
SRCS = $(wildcard src/*.c)
OBJS = $(SRCS:.c=.o)

# Test source files and targets
TEST_SRCS = $(wildcard tests/*.c)
TEST_BINS = $(TEST_SRCS:tests/%.c=%)

# Main target
TARGET = transformer

all: $(TARGET)

$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Rule for building each test
# Tests need to be linked with all src objects
%: tests/%.c $(OBJS)
	$(CC) $(CFLAGS) $< $(OBJS) -o $@ $(LDFLAGS)

# Build all tests
tests: $(TEST_BINS)

# Run all tests
run-tests: tests
	@for test in $(TEST_BINS); do \
		echo "Running $$test..."; \
		./$$test || echo "$$test failed with exit code $$?"; \
		echo "--------------------"; \
	done

# Debug with AddressSanitizer
asan: CFLAGS += -fsanitize=address -g
asan: LDFLAGS += -fsanitize=address
asan: clean run-tests

clean:
	rm -f src/*.o main.o $(TARGET) $(TEST_BINS)

.PHONY: all clean tests run-tests asan
