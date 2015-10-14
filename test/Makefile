NVCC := nvcc
NVFLAGS := -I./ -I/usr/local/cuda-7.0/include -L/usr/local/cuda-7.0/lib64 -I../include/common -I../include/mul -L../build/lib
NVLIBS := -lcudart -lmatrix.o

DIR_SRCS := ../src/mul

OBJS := sample_matrix_mul.o

TARGETS := test_matrix_mul

$(TARGETS) : $(OBJS)
	$(NVCC) $(NVFLAGS) -o $@ $< $(NVLIBS)

sample_matrix_mul.o : sample_matrix_mul.cu
	$(NVCC) $(NVFLAGS) -c $<

.PHONY: clean
clean :
	rm -f sample_matrix_mul *.o
