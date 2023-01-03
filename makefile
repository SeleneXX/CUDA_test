target1 := ./vectoradd
target2 := ./matmul
target3 := ./tileddot
source1 = vectoradd.cu
source2 = MatrixMultiplication.cu
source3 = Tileddot.cu

all: $(target1) $(target2) $(target3)

$(target1):$(source1)
	nvcc $(source1) -o $(target1)

$(target2):$(source2)
	nvcc $(source2) -o $(target2)

$(target3):$(source3)
	nvcc $(source3) -o $(target3)

.PHONY: clean
clean:
	-rm -rf $(target1) $(target2) $(target3)

