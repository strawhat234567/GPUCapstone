NVCC ?= nvcc
TARGET := attention_bench
CXXFLAGS := -O3 -std=c++17 -lineinfo

SRC := \
	main.cu \
	attention_naive.cu \
	attention_tiled.cu \
	attention_online_tiled.cu \
	cpu_reference.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET) --n 128 --d 64 --iters 100 --mode all

clean:
	rm -f $(TARGET)