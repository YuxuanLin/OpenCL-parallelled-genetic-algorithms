#include "cl_generation.h"
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
int main() {
    //------------------------------------------------------------
    //                     STEP 1: GA Host prep
    //------------------------------------------------------------

    int popsize, lchrom, maxgen, gen,nvar;
    unsigned long ncross, nmutation;
    double max,min,avg,pcross,pmutation,sumfitness,rvar[2*Nvar+1];
    ind_t oldpop[Maxpop+1], newpop[Maxpop+1],outpop[Maxpop+1];
    max_t max_ult;
    time_t t_start, t_finish, t_diff;
    time(&t_start);
    max_ult.gen = 1;
    max_ult.objective = 0.;  /*initialize for maximum value*/
    gen = 0;
    initialize(&popsize,&lchrom,&maxgen,&pcross,&pmutation,oldpop,&max,&min,&avg,&sumfitness,&nmutation,&ncross,&nvar,rvar,&max_ult,gen);

    if(fabs(max-min) <= Tol)//initialization check
    {
        printf("Reduce the range of the variables in SGA3.VAR\n");
        exit(1);
    }
    //------------------------------------------------------------
    //                     STEP 2: OpenCL Host prep
    //------------------------------------------------------------
    size_t sizePop = sizeof(ind_t)*(Maxpop+1);
    size_t sizeNcross = sizeof(unsigned long);
    size_t sizeNmutation  = sizeof(unsigned long);
    size_t sizeRvar = sizeof(double) * (2*Nvar+1);
    cl_int status;  
    //------------------------------------------------------------
    //      STEP 3: Discover and initialize the platforms
    //------------------------------------------------------------
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;
    CL_CHECK(clGetPlatformIDs(0, NULL, &numPlatforms));
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    CL_CHECK(clGetPlatformIDs(numPlatforms, platforms, NULL));
    //------------------------------------------------------------
    //      STEP 4: Discover and initialize the devices
    //------------------------------------------------------------
    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;
    CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices));
    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));
    cl_uint devices_n = 2;
    CL_CHECK(clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL,numDevices,devices, &devices_n));
    //Choose device in devices array: 0: CPU 1:Iris GPU 2:GTX750 GPU
    int deviceIdx=0;
    //Set up information prompt
    char buffer[10240];
    cl_uint buf_uint;
    cl_ulong buf_ulong;
    cl_device_fp_config cfg;
    //Prompt chosen device information
    CL_CHECK(clGetDeviceInfo(devices[deviceIdx], CL_DEVICE_NAME, sizeof(buffer), buffer, NULL));
    printf("  DEVICE_NAME = %s\n", buffer);
    CL_CHECK(clGetDeviceInfo(devices[deviceIdx], CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL));
    printf("  DEVICE_VENDOR = %s\n", buffer);
    CL_CHECK(clGetDeviceInfo(devices[deviceIdx], CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL));
    printf("  DEVICE_VERSION = %s\n", buffer);
    CL_CHECK(clGetDeviceInfo(devices[deviceIdx], CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL));
    printf("  DRIVER_VERSION = %s\n", buffer);
    CL_CHECK(clGetDeviceInfo(devices[deviceIdx], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(buf_uint), &buf_uint, NULL));
    printf("  DEVICE_MAX_COMPUTE_UNITS = %u\n", (unsigned int)buf_uint);
    CL_CHECK(clGetDeviceInfo(devices[deviceIdx], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(buf_uint), &buf_uint, NULL));
    printf("  DEVICE_MAX_CLOCK_FREQUENCY = %u\n", (unsigned int)buf_uint);
    CL_CHECK(clGetDeviceInfo(devices[deviceIdx], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(buf_ulong), &buf_ulong, NULL));
    printf("  DEVICE_GLOBAL_MEM_SIZE = %llu\n", (unsigned long long)buf_ulong);
    clGetDeviceInfo(*devices, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cfg), &cfg, NULL);
    printf("  DEVICE_DOUBLE_FP_CONFIG = %llu\n", cfg);

    //------------------------------------------------------------
    //      STEP 5: Create a context
    //------------------------------------------------------------
    cl_context context = NULL;
    context = (clCreateContext(NULL,numDevices,devices,NULL,NULL,&status));

    //------------------------------------------------------------
    //      STEP 6: Create a command queue
    //------------------------------------------------------------
    cl_command_queue cmdQueue;
    cmdQueue = clCreateCommandQueue(context, devices[deviceIdx], 0, &status);

    //------------------------------------------------------------
    //                   STEP 7: Read kernel file
    //------------------------------------------------------------
    FILE *fp;
    char fileName[] = "./kernelGeneration.cl";
    char *source_str;
    size_t source_size;
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    //-----------------------------------------------------
    //      STEP 8: Create and compile the program
    //-----------------------------------------------------
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str,(const size_t *)&source_size, &status);
    clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    size_t log_size;
    clGetProgramBuildInfo(program, devices[deviceIdx], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *log = (char *) malloc(log_size);
    clGetProgramBuildInfo(program, devices[deviceIdx], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    printf("%s\n", log);

    //-----------------------------------------------------
    //      STEP 9: Create the kernel
    //----------------------------------------------------- 
    cl_kernel kernel = NULL;
    kernel = clCreateKernel(program, "generation", &status);
    //-----------------------------------------------------
    //      STEP 10: Configure the work-item structure
    //----------------------------------------------------- 
    size_t globalWorkSize[1];    
    // There are 'elements' work-items 
    globalWorkSize[0] = 50 ;
    //------------------------------------------------------------
    //      STEP 11: Create device buffers
    //------------------------------------------------------------
    cl_mem bfoldpop;  
    cl_mem bfnewpop;  
    cl_mem bfncross;
    cl_mem bfnmutation;
    cl_mem bfRvar;
    cl_mem bfRandomSets;
    cl_mem bfDebug;
    int numRandomSets = 10000;
    size_t sizeRandomSets = sizeof(double) * numRandomSets;

    int numDebug = 100000;
    size_t  sizeDebug = sizeof(double) *numDebug;
    bfoldpop = clCreateBuffer(context, CL_MEM_READ_WRITE,sizePop, NULL, &status);
    bfnewpop = clCreateBuffer(context, CL_MEM_READ_WRITE,sizePop, NULL, &status);
    bfncross = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeNcross, NULL, &status);
    bfnmutation = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeNmutation, NULL, &status);
    bfRvar = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeRvar, NULL, &status);
    bfRandomSets = clCreateBuffer(context, CL_MEM_READ_WRITE,sizeRandomSets, NULL, &status);
    bfDebug = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeDebug,NULL,&status);
    
    //----------------------------------------------
    //         STEP 12: GA loop begin
    //----------------------------------------------
    do{
        gen++; 

        printf("n.gen = %d",gen);
        printf("\n\033[F\033[J");

        //calculate sumfitness
        scalepop(popsize,max,avg,min,&sumfitness,oldpop);
        //generate random number sets, because openCL kernel doesn't support random number generation
        double *randomSets;
        randomSets = (double *)malloc(sizeRandomSets);
        for(int i = 0; i < numRandomSets; i++){
            randomSets[i] = (double) rand()/RAND_MAX;
        }
        //----------------OpenCL Merge in-------------------------
        //      STEP 13: Write host data to device buffers
        //----------------------------------------------------- 
        CL_CHECK( clEnqueueWriteBuffer(cmdQueue, bfoldpop, CL_TRUE, 0, sizePop,(const void *)oldpop, 0, NULL, NULL));
        CL_CHECK( clEnqueueWriteBuffer(cmdQueue, bfnewpop, CL_TRUE, 0, sizePop,(const void *)newpop, 0,NULL, NULL));
        CL_CHECK( clEnqueueWriteBuffer(cmdQueue, bfncross, CL_TRUE, 0, sizeNcross,(const void *)&ncross, 0, NULL, NULL));
        CL_CHECK( clEnqueueWriteBuffer(cmdQueue, bfnmutation, CL_TRUE, 0, sizeNmutation,(const void *)&nmutation, 0, NULL, NULL));     
        CL_CHECK( clEnqueueWriteBuffer(cmdQueue, bfRvar, CL_TRUE, 0, sizeRvar,(const void *)&rvar, 0, NULL, NULL));  
        CL_CHECK( clEnqueueWriteBuffer(cmdQueue, bfRandomSets, CL_TRUE, 0, sizeRandomSets,(const void *)randomSets, 0, NULL, NULL));    
        //-----------------------------------------------------
        //       STEP 14: Set the kernel arguments
        //----------------------------------------------------- 
        CL_CHECK(clSetKernelArg(kernel, 0, sizeof(int), &popsize));//popsize
        CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bfoldpop));//oldpop
        CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bfnewpop));//newpop
        CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &lchrom));//lchrom
        CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &bfncross));//ncross
        CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &bfnmutation));//nmutation
        CL_CHECK(clSetKernelArg(kernel, 6, sizeof(double), &pcross));//pcross
        CL_CHECK(clSetKernelArg(kernel, 7, sizeof(double), &pmutation));//pmutation
        CL_CHECK(clSetKernelArg(kernel, 8, sizeof(int), &nvar));//nvar
        CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), &bfRvar));//rvar
        CL_CHECK(clSetKernelArg(kernel, 10, sizeof(double), &avg));//avg
        CL_CHECK(clSetKernelArg(kernel, 11, sizeof(cl_mem), &bfRandomSets));//randomsets
        CL_CHECK(clSetKernelArg(kernel, 12, sizeof(cl_mem), &bfDebug));//randomsets

        //-----------------------------------------------------
        //      STEP 15: Enqueue the kernel for execution
        //----------------------------------------------------- 
        CL_CHECK(clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL));//Replace the C code "generation()"

        //-----------------------------------------------------
        //      STEP 16: Read the output buffer back to the host
        //----------------------------------------------------- 
        // double debug[100000] ;
        CL_CHECK(clEnqueueReadBuffer(cmdQueue, bfoldpop, CL_TRUE, 0, sizePop, oldpop, 0, NULL, NULL));//oldpop
        CL_CHECK(clEnqueueReadBuffer(cmdQueue, bfnewpop, CL_TRUE, 0, sizePop, newpop, 0, NULL, NULL));//newpop
        CL_CHECK(clEnqueueReadBuffer(cmdQueue, bfncross, CL_TRUE, 0, sizeNcross, &ncross, 0, NULL, NULL));//ncross
        CL_CHECK(clEnqueueReadBuffer(cmdQueue, bfnmutation, CL_TRUE, 0, sizeNmutation, &nmutation, 0, NULL, NULL));//nmutation
        // for(int i = 0; i < 100; i++)
        //     printf("%G\n",debug[i]);
        //----------------OpenCL Merge End-------------------------
        statistics(popsize,&max,&avg,&min,&sumfitness,newpop,&max_ult,nvar,gen);

        #if SHORT
        if(gen == 1 || gen == maxgen+1)
        #endif

        report(gen,oldpop,newpop,lchrom,max,avg,min,sumfitness,
            nmutation,ncross,popsize,nvar,&max_ult);
        memcpy(oldpop, newpop, sizeof(newpop));
    } while(gen <= maxgen);

    //-----------------------------------------------------
    //                 STEP 17: GA finish
    //-----------------------------------------------------
    time(&t_finish);  
    t_diff = t_finish-t_start;
    printf("Total Time (mins.) = %f\n", (float) t_diff/60.);
    printf("The results are stored in ***genout.dat***\n");

    //-----------------------------------------------------
    //      STEP 18: Release OpenCL resources
    //----------------------------------------------------- 
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bfnewpop);
    clReleaseMemObject(bfnmutation);
    clReleaseMemObject(bfncross);
    clReleaseMemObject(bfoldpop);
    clReleaseMemObject(bfRvar);
    clReleaseMemObject(bfRandomSets);
    clReleaseContext(context);

    // Free host resources
    free(source_str);
    free(platforms);
    free(devices);
}

