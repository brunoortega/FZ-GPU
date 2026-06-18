main: src/fz.cu
	nvcc src/claunch_cuda.cu -Iinclude --extended-lambda -c
	nvcc src/fz.cu claunch_cuda.o -o fz-gpu

isolated: src/fz_isolated.cu
	nvcc src/claunch_cuda.cu -Iinclude --extended-lambda -c
	nvcc src/fz_isolated.cu claunch_cuda.o -o fz-gpu-isolated

clean:
	rm -rf fz-gpu fz-gpu-isolated claunch_cuda.o
