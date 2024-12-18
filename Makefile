# 컴파일러 및 플래그 설정
CXX = clang++
METAL = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib
CXXFLAGS = -framework Metal -framework Foundation -framework CoreGraphics -I./include
BUILD_DIR = build
SRC_DIR = src

# 소스 파일들
SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/compare.cpp $(SRC_DIR)/cnn_metal.mm $(SRC_DIR)/cnn_seq.cpp
METAL_SRC = $(SRC_DIR)/shader.metal

# 메탈 라이브러리
METAL_AIR = $(BUILD_DIR)/shader.air
METAL_LIB = $(BUILD_DIR)/shader.metallib

# 최종 실행 파일
TARGET = $(BUILD_DIR)/program

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR)

# Metal 셰이더 컴파일
$(METAL_AIR): $(METAL_SRC)
	$(METAL) -c $< -o $@

$(METAL_LIB): $(METAL_AIR)
	$(METALLIB) $< -o $@

# 최종 링크
$(TARGET): $(METAL_LIB) $(SRCS)
	$(CXX) $(SRCS) $(CXXFLAGS) -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean dirs
