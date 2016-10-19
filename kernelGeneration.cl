#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define Maxpop 10000+1
typedef int chromosome;
typedef struct{
                chromosome chrom[11+1];
                double x[11+1],objective,fitness;
                int parent1, parent2;
                int xsite;
              } ind_t;
double chrom_map_x(int chrom,double range1,double range2,int lchrom);
void bit_on_off(__global int *chrom, int i);
void bit_on(__global int *chrom, int i);
int bit_check(int chrom, int i);
int flip(double probability, double * randomSet, int *randomSetIdx);
double objfunc( double x1, double x2, double x3, int nvar);
void crossover(__global int *parent1,__global int *parent2,__global int *child1,__global int *child2,
               int lchrom,__global unsigned long *ncross, __global unsigned long *nmutation,int *jcross,
               double pcross,double pmutation, int nvar,  double *randomSet, int * randomSetIdx);//
int round_(int j, int lchrom);
int  rnd(int low,int high, double random);
int  mutation(double pmutation,  unsigned long nmutation,  double *randomSet, int *randomSetIdx);
int select1(int *nremain,int *choices, double *randomSet, int *ranodmSetIdx);
void pre_select1(int popsize, double avg, __global ind_t *pop,  int *choices, double *randomSet, int *randomSetIdx);//
__kernel void generation( int popsize,__global ind_t *oldpop,__global ind_t *newpop,int lchrom,__global unsigned long *ncross,__global unsigned long *nmutation,double pcross,double pmutation,int nvar, __global double *rvar,double avg,__global double *randomSet,__global double *debug)
{
    int k, kk;
    int j, mate1, mate2, jcross,nremain, choices[Maxpop+1];
    double localRandomSet[500];
    j = get_global_id(0);    
    int randomSetIdx = 0;

    for (int i = 0; i <  200; i++) 
        localRandomSet[i] = randomSet[(j)* 200 + i];
    j *= 2;
    j += 1;
    nremain = popsize;
    pre_select1(popsize, avg, oldpop, choices,localRandomSet,&randomSetIdx);

    mate1 = select1(&nremain, choices,localRandomSet,&randomSetIdx);
    mate2 = select1(&nremain, choices,localRandomSet,&randomSetIdx);//选择双亲
    
    // //j 计数问题
    crossover(oldpop[mate1].chrom, oldpop[mate2].chrom,
        &(newpop[j].chrom[0]), &(newpop[j+1].chrom[0]), 
        lchrom,ncross,nmutation,&jcross,pcross,pmutation,nvar,localRandomSet,&randomSetIdx);
 
    for(k = 1; k <= nvar; k++)
    {
        kk = 2*k - 1;
        newpop[j].x[k] = chrom_map_x(newpop[j].chrom[k],rvar[kk],rvar[kk+1],lchrom);
    }

    //calculate raw fitness
    newpop[j].objective = objfunc(newpop[j].x[1],newpop[j].x[2],newpop[j].x[3],nvar);
    newpop[j].parent1 = mate1;
    newpop[j].parent2 = mate2;

    newpop[j].xsite = jcross;
    
    for(k = 1; k <= nvar; k++)
    {
        kk = 2*k - 1;
        newpop[j+1].x[k] = chrom_map_x(newpop[j+1].chrom[k],rvar[kk],rvar[kk+1],lchrom);
    }
    newpop[j+1].objective = objfunc(newpop[j+1].x[1],newpop[j+1].x[2],newpop[j].x[3], nvar);
    newpop[j+1].parent1 = mate1;
    newpop[j+1].parent2 = mate2;
    newpop[j+1].xsite = jcross;

}
double objfunc( double x1, double x2, double x3, int nvar) //fitness 计算函数
{
    double result,penalty;
    result = 700 - (x1+pow(x2,2.)+x2*x3);
    penalty =1000 * pow(x1+x2+x3,2.) ;
    result -= penalty;
    if(fabs(result) <= 0. ) result = 1.e-30;
    return(result);
}
double chrom_map_x(int chrom,double range1,double range2,int lchrom)
{
    double diff,add_percent,res;

    diff = range2-range1;
    add_percent = ((double) chrom/ (pow(2.0f,(float)lchrom)-1.))*diff;
    res = range1 + add_percent;

    return(res);
}
void crossover(__global int *parent1,__global int *parent2,__global int *child1,__global int *child2,
               int lchrom,__global unsigned long *ncross, __global unsigned long *nmutation,int *jcross,
               double pcross,double pmutation, int nvar,  double *randomSet, int * randomSetIdx)
{
    int k,kk;
    int j, lighted, test, rn;


    // memcpy(child1, parent1, sizeof(int)*(nvar+1));
    // memcpy(child2, parent2, sizeof(int)*(nvar+1));
    for ( int i = 0; i < (nvar + 1); i++){
        child1[i] = parent1[i];
        child2[i] = parent2[i];
    }
    //pcross 是交叉的概率
    if(flip(pcross,randomSet,randomSetIdx)  == 1)
    {
        //交叉执行，计算交叉位数jcross
        *jcross = rnd(1, nvar*lchrom-1,randomSet[*randomSetIdx]);
        *randomSetIdx += 1;
        //交叉总数自增
        *ncross += 1;
    }
    else
        *jcross = nvar*lchrom;

    rn = 0;  
    /*chrom counter*/
    kk = 1;
    //cross位数确定：从1到jcross
    /*
     * 变异
     */
    for(j = 1; j <= *jcross; j++)
    {

        if(rn == 1) kk++;

        rn = round_(j, lchrom);

        k = j - (kk-1)*lchrom;
        //判断是否变异
        test = mutation(pmutation, *nmutation,randomSet,randomSetIdx);
        /*test = [0] no change , test = 1 bit changed kth bit is altered*/
        if(test == 1) bit_on_off(&child1[kk],k); /* mutation */

        test = mutation(pmutation, *nmutation,randomSet,randomSetIdx);
         /*test = [0] no change , test = 1 bit changed kth bit is altered*/
        if(test == 1)bit_on_off(&child2[kk],k);/* mutation */
        k++;
    }

    if(*jcross != nvar*lchrom)
    {
        for(j = *jcross+1; j <= nvar*lchrom; j++)
        {
            if(rn == 1) kk++;
            rn = round_(j, lchrom);
            k = j - (kk-1)*lchrom;
            lighted = bit_check(parent2[kk],k);        /*lighted = [1] if bit is on */
            test = mutation(pmutation, *nmutation,randomSet,randomSetIdx);
                   /*test = [0] no change , test = 1 bit changed jth bit is altered*/
            bit_on(&child1[kk],k);
            if(lighted == 0) bit_on_off(&child1[kk],k);
            if(test == 1)bit_on_off(&child1[kk],k);    /* mutate */

            lighted = bit_check(parent1[kk],k);        /*lighted = [1] if bit is on */
            test = mutation(pmutation, *nmutation,randomSet,randomSetIdx);
                   /*test = [0] no change , test = 1 bit changed jth bit is altered*/
            bit_on(&child2[kk],k);
            if(lighted == 0) bit_on_off(&child2[kk],k);
            if(test == 1)bit_on_off(&child2[kk],k);    /* mutate */
        }
    }
}

int select1(int *nremain,int *choices, double *randomSet, int *randomSetIdx)
{
    int jpick, index;

    jpick = rnd(1, *nremain,randomSet[*randomSetIdx]);
    *randomSetIdx += 1;
    index = choices[jpick];

    choices[jpick] = choices[*nremain];

    *nremain = *nremain - 1;

    return(index);
}

void pre_select1(int popsize, double avg, __global ind_t *pop, int *choices, double *randomSet, int *randomSetIdx)
{
    //choices output
    int j, k;
    int jassign, winner;
    double expected, fraction[Maxpop+1], whole;

    j = 0; k = 0;

    do{
        j++;
        expected = pop[j].fitness/(avg+1.e-30);

        fraction[j] = modf(expected, &whole);

        jassign = (int) whole;

        while(jassign > 0)
        {
            k++; jassign--;
            choices[k] = j;
        }
    }
    while(j != popsize);
    
    j = 0;
    while(k < popsize)
    {
        j++;
        if(j > popsize) j = 1;
        if(fraction[j] > 0.0)
        {
            winner = flip(fraction[j],randomSet,randomSetIdx);
            if(winner == 1)
            {
                k++;
                choices[k]= j;
                (fraction[j])--;
            }
        }
    }
}
int flip(double probability,  double * randomSet, int *randomSetIdx)
{
    double random1_t;
    if(probability == 1.0) return(1);

    random1_t = randomSet[*randomSetIdx];
    *randomSetIdx += 1;
    // random1_t = 0.4;
    if(random1_t <= probability) return(1);
    else return(0);
}



int  rnd(int low,int high, double random)
{
    int i;

    if(low >= high) i = low;
    else
    {
        i = (int) (random*(high-low+1) + low);
        if(i > high) i = high;
    }
    return(i);
}


int  mutation(double pmutation,  unsigned long nmutation,  double *randomSet, int *randomSetIdx)
{
    int mutate;

    mutate = flip(pmutation,randomSet,randomSetIdx);
    if(mutate == 1)
    {
        nmutation += 1;
        return(1);
    }
    else
        return(0);
}

int round_(int j, int lchrom)
{

    if(fmod((float)j,(float)lchrom) == 0)
    return(1);

    return(0);
}



int  bit_check(int chrom, int i)
{
    switch(i){
        case 1:  if(chrom & 01)return(1); break; /*if bit was on return [1]*/
        case 2:  if(chrom & 02)return(1); break;
        case 3:  if(chrom & 04)return(1); break;
        case 4:  if(chrom & 010)return(1); break;
        case 5:  if(chrom & 020)return(1); break;
        case 6:  if(chrom & 040)return(1); break;
        case 7:  if(chrom & 0100)return(1); break;
        case 8:  if(chrom & 0200)return(1); break;
        case 9:  if(chrom & 0400)return(1); break;
        case 10: if(chrom & 01000)return(1); break;
        case 11: if(chrom & 02000)return(1); break;
        case 12: if(chrom & 04000)return(1); break;
        case 13: if(chrom & 010000)return(1); break;
        case 14: if(chrom & 020000)return(1); break;
        case 15: if(chrom & 040000)return(1); break;
        case 16: if(chrom & 0100000)return(1); break;
        case 17: if(chrom & 0200000)return(1); break;
        case 18: if(chrom & 0400000)return(1); break;
        case 19: if(chrom & 01000000)return(1); break;
        case 20: if(chrom & 02000000)return(1); break;
        case 21: if(chrom & 04000000)return(1); break;
        case 22: if(chrom & 010000000)return(1); break;
        case 23: if(chrom & 020000000)return(1); break;
        case 24: if(chrom & 040000000)return(1); break;
        case 25: if(chrom & 0100000000)return(1); break;
        case 26: if(chrom & 0200000000)return(1); break;
        case 27: if(chrom & 0400000000)return(1); break;
        case 28: if(chrom & 01000000000)return(1); break;
        case 29: if(chrom & 02000000000)return(1); break;
        case 30: if(chrom & 04000000000)return(1); break;
        case 31: if(chrom & 010000000000)return(1); break;
    }
    return(0);   /* if bit is no on then return [0] */
}



void bit_on_off(__global int *chrom, int i)
{
    switch(i){
        case 1:  *chrom ^= 01; break;  /*set on the 1st bit */
        case 2:  *chrom ^= 02; break;
        case 3:  *chrom ^= 04; break;
        case 4:  *chrom ^= 010; break;
        case 5:  *chrom ^= 020; break;
        case 6:  *chrom ^= 040; break;
        case 7:  *chrom ^= 0100; break;
        case 8:  *chrom ^= 0200; break;
        case 9:  *chrom ^= 0400; break;
        case 10: *chrom ^= 01000; break;
        case 11: *chrom ^= 02000; break;
        case 12: *chrom ^= 04000; break;
        case 13: *chrom ^= 010000; break;
        case 14: *chrom ^= 020000; break;
        case 15: *chrom ^= 040000; break;
        case 16: *chrom ^= 0100000; break;
        case 17: *chrom ^= 0200000; break;
        case 18: *chrom ^= 0400000; break;
        case 19: *chrom ^= 01000000; break;
        case 20: *chrom ^= 02000000; break;
        case 21: *chrom ^= 04000000; break;
        case 22: *chrom ^= 010000000; break;
        case 23: *chrom ^= 020000000; break;
        case 24: *chrom ^= 040000000; break;
        case 25: *chrom ^= 0100000000; break;
        case 26: *chrom ^= 0200000000; break;
        case 27: *chrom ^= 0400000000; break;
        case 28: *chrom ^= 01000000000; break;
        case 29: *chrom ^= 02000000000; break;
        case 30: *chrom ^= 04000000000; break;
        case 31: *chrom ^= 010000000000; break;
    }
}

void bit_on(__global int *chrom, int i)
{
    switch(i){
        case 1:  *chrom |= 01; break;  /*set on the 1st bit */
        case 2:  *chrom |= 02; break;
        case 3:  *chrom |= 04; break;
        case 4:  *chrom |= 010; break;
        case 5:  *chrom |= 020; break;
        case 6:  *chrom |= 040; break;
        case 7:  *chrom |= 0100; break;
        case 8:  *chrom |= 0200; break;
        case 9:  *chrom |= 0400; break;
        case 10: *chrom |= 01000; break;
        case 11: *chrom |= 02000; break;
        case 12: *chrom |= 04000; break;
        case 13: *chrom |= 010000; break;
        case 14: *chrom |= 020000; break;
        case 15: *chrom |= 040000; break;
        case 16: *chrom |= 0100000; break;
        case 17: *chrom |= 0200000; break;
        case 18: *chrom |= 0400000; break;
        case 19: *chrom |= 01000000; break;
        case 20: *chrom |= 02000000; break;
        case 21: *chrom |= 04000000; break;
        case 22: *chrom |= 010000000; break;
        case 23: *chrom |= 020000000; break;
        case 24: *chrom |= 040000000; break;
        case 25: *chrom |= 0100000000; break;
        case 26: *chrom |= 0200000000; break;
        case 27: *chrom |= 0400000000; break;
        case 28: *chrom |= 01000000000; break;
        case 29: *chrom |= 02000000000; break;
        case 30: *chrom |= 04000000000; break;
        case 31: *chrom |= 010000000000; break;
    }
}
