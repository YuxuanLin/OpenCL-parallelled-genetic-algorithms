#ifndef __cl_GENERATION_H__
#define __cl_GENERATION_H__
#define Choice 1
#define  SHORT 1
#define SunOs  0
#define Tol2 1.e-30;
#define Maxpop 10000+1
#define Nvar   10+1
#define Maxchrom 16
#define Fmultiple 2
#define Tol (pow((10.),(-30.)))
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define MAX_SOURCE_SIZE (0x100000)
#define CL_CHECK(_expr)                                                         \
   do {                                                                         \
     cl_int _err = _expr;                                                       \
     if (_err == CL_SUCCESS)                                                    \
       break;                                                                   \
     fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     abort();                                                                   \
   } while (0)
typedef int chromosome;

typedef struct{
              chromosome chrom[Nvar+1];
              double x[Nvar+1],objective,fitness;
              int parent1, parent2;
              int xsite;
              } ind_t;

typedef struct{
              int gen;
              chromosome chrom[Nvar+1];
              double x[Nvar+1],objective;
              } max_t;

void initialize(int *,int *,int *,double *,double *,
          ind_t *,double *,double *,double *,double *,unsigned long *,unsigned long *,
          int *, double *,max_t *max_ult,int);

void initpop(int popsize,ind_t *oldpop,int lchrom,int nvar,
                 double *rvar);
double chrom_map_x(int chrom,double range1,double range2,int lchrom);
void bit_on_off(int *chrom, int i);
void bit_on(int *chrom, int i);
int bit_check(int chrom, int i);
int flip(double probability);
double random1(void);
double objfunc(double *x,int nvar);
void initdata(int *popsize,int *lchrom,int *maxgen,double *pcross,
          double *pmutation,unsigned long *nmutation,unsigned long *ncross,int *nvar,
          double *rvar);

void initreport(int popsize,int lchrom,int maxgen,double pcross,
                double pmutation,double max,double avg,double min,
                double sumfitness);

void generation(int popsize,double *sumfitness,ind_t *oldpop,
                     ind_t *newpop,int lchrom,unsigned long *ncross,unsigned long *nmutation,
                     double pcross,double pmutation,int nvar,double *rvar,
                     double avg);

void statistics(int popsize,double *max,double *avg,double *min,
              double *sumfitness,ind_t*newpop,max_t *max_ult,int nvar,int gen);
int ga_select(int popsize, double sumfitness, ind_t *old);

void pre_select1(int popsize, double avg, ind_t *pop, int *choices);

int select1(int *nremain,int *choices);

void crossover(int *parent1,int *parent2,int *child1,int *child2,
          int lchrom,unsigned long *ncross,unsigned long *nmutation,int *jcross,
          double pcross,double pmutation,int nvar);

int round_(int j, int lchrom);

int rnd(int low,int high);

int mutation(double pmutation,unsigned long *nmutation);

void report(int gen,ind_t *oldpop,ind_t *newpop,
          int lchrom,double max,double avg,double min,double sumfitness,
          unsigned long nmutation,unsigned long ncross, int popsize,
          int nvar,max_t *max_ult);

void scalepop(int popsize, double max, double avg, double min,
              double *sumfitness,ind_t *pop);

void prescale(double umax, double uavg, double umin, double *a, double *b);


void initialize(int *popsize,int *lchrom,int *maxgen,double *pcross,
       double  *pmutation,ind_t *oldpop,double *max,
       double *min,double *avg,double *sumfitness,unsigned long *nmutation,
       unsigned long *ncross, int *nvar, double *rvar,max_t *max_ult,int gen)
{
    initdata(popsize,lchrom,maxgen,pcross,pmutation,nmutation,ncross,
          nvar, rvar);

    initpop(*popsize,oldpop,*lchrom,*nvar,rvar);

    statistics(*popsize,max,avg,min,sumfitness,oldpop,max_ult,*nvar,gen);

    initreport(*popsize,*lchrom,*maxgen,*pcross,*pmutation,
             *max,*avg,*min,*sumfitness);
}

void initdata(int *popsize,int *lchrom,int *maxgen,double *pcross,
              double *pmutation,unsigned long *nmutation,unsigned long *ncross,int *nvar,
              double *rvar)
{
    register int i,j;
    FILE *fp;
    char buf[120], dummy[50];
    unsigned seed;

    if((fp = fopen("sga3.var","r")) == (FILE *)NULL)
    {
        printf("something wrong with sga3.var\n");
        exit(1);
    }

    fgets(buf,120,fp); 
    sscanf(buf,"%s %d",dummy,popsize);

    fgets(buf,120,fp); 
    sscanf(buf,"%s %d",dummy,lchrom);

    fgets(buf,120,fp); 
    sscanf(buf,"%s %d",dummy,maxgen);

    fgets(buf,120,fp); 
    sscanf(buf,"%s %lf",dummy,pcross);

    fgets(buf,120,fp); 
    sscanf(buf,"%s %lf",dummy,pmutation);

    fgets(buf,120,fp); 
    sscanf(buf,"%s %d",dummy,&seed);

    fgets(buf,120,fp); 
    sscanf(buf,"%s %d",dummy,nvar);

    for(i = 1; i <= *nvar; i++)
    {
        j = 2*i-1;
        fgets(buf,120,fp);
        sscanf(buf,"%s %lf %s %lf",dummy,&rvar[j],dummy,&rvar[j+1]);
    }

    srand(seed);

    *nmutation = 0;

    *ncross = 0;

    fclose(fp);
}


void initpop(int popsize,ind_t *oldpop,int lchrom,int nvar,double *rvar)
{
    register int k,kk;
    int j, j1, flip_t;

    for(j = 1; j <= popsize; j++)  /* zero lchrom*/                          
        for(k = 1; k <= nvar; oldpop[j].chrom[k++] = 0);

    for(j = 1; j <= popsize; j++)
    {
        for(k = 1; k <= nvar; k++)
            for(j1 = 1; j1 <= lchrom; j1++)
            {
                flip_t = flip(0.5);
                if(flip_t > 0) bit_on(&(oldpop[j].chrom[k]), j1);
            }


        for(k = 1; k <= nvar; k++)
        {
            kk = 2*k - 1;
            oldpop[j].x[k] = chrom_map_x(oldpop[j].chrom[k],
                                   rvar[kk],rvar[kk+1],lchrom);
        }

        oldpop[j].objective = objfunc(oldpop[j].x,nvar);

        oldpop[j].parent1 = oldpop[j].parent2= oldpop[j].xsite = 0;

    }
}

double chrom_map_x(int chrom,double range1,double range2,int lchrom)
{
    double diff,add_percent,res;

    diff = range2-range1;
    add_percent = ((double) chrom/ (pow(2.,(double)lchrom)-1.))*diff;
    res = range1 + add_percent;

    return(res);
}

int flip(double probability)
{
    double random1_t;
    if(probability == 1.0) return(1);

    random1_t = random1();

    if(random1_t <= probability) return(1);
    else return(0);
}

double random1(void)
{
    double result;
    result = (double) rand()/RAND_MAX;
    return(result);
}

void bit_on_off(int *chrom, int i)
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

void bit_on(int *chrom, int i)
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


void statistics(int popsize,double *max,double *avg,double *min,
                 double *sumfitness,ind_t *pop,max_t *max_ult,int nvar,int gen)
{
    int j,jmax;

    jmax = 1;

    *sumfitness = pop[1].objective;
    *min        = pop[1].objective;
    *max        = pop[1].objective;

    for(j = 2; j <= popsize; j++)
    {
        *sumfitness += pop[j].objective;
        if(pop[j].objective > *max)
        {
            *max = pop[j].objective;
            jmax = j;
        }

        if(pop[j].objective < *min) *min = pop[j].objective;
    }

    if(*max > max_ult->objective)
    {
        max_ult->objective = *max;
        max_ult->gen = gen;
        memcpy(&(max_ult->chrom[0]),&(pop[jmax].chrom[0]), sizeof(chromosome)*(nvar+1));
        memcpy(&(max_ult->x[0]),&(pop[jmax].x[0]), sizeof(double)*(nvar+1));
    }


    *avg = *sumfitness/((double)popsize);
}

void initreport(int popsize,int lchrom,int maxgen,double pcross,
                double pmutation,double max,double avg,double min,
                double sumfitness)
{
    FILE *fpout;

    if((fpout=fopen("genout.dat","w")) == (FILE *) NULL)
    {
        printf("cannot open genout.dat\n");
        exit(1);
    }

    fprintf(fpout,"Population size (popsize) = %5d\n",popsize);
    fprintf(fpout,"Chromosome length (lchrom) = %5d\n",lchrom);
    fprintf(fpout,"Maximum # of generations (maxgen) %5d\n",maxgen);
    fprintf(fpout,"Crossover probability (pcross) = %10.5e\n",pcross);
    fprintf(fpout,"Mutation probability (pmutation) = %10.5e\n",pmutation);
    fprintf(fpout,"\n Initial Generation Statistics\n");
    fprintf(fpout,"---------------------------------\n");
    fprintf(fpout,"\n");
    fprintf(fpout,"Initial population maximum fitness = %10.5e\n", max);
    fprintf(fpout,"Initial population average fitness = %10.5e\n",avg);
    fprintf(fpout,"Initial population minimum fitness = %10.5e\n",min);
    fprintf(fpout,"Initial population sum of  fitness = %10.5e\n",sumfitness);
    fprintf(fpout,"\n\n\n");
    fclose(fpout);
}

void generation(int popsize,double *sumfitness,ind_t *oldpop,
     ind_t *newpop,int lchrom,unsigned long *ncross,unsigned long *nmutation,
     double pcross,double pmutation,int nvar, double *rvar,double avg)
{
    register int k, kk;
    int j, mate1, mate2, jcross,nremain, choices[Maxpop+1];

    j = 1;

    #if Choice
    nremain = popsize;
    pre_select1(popsize, avg, oldpop, choices);
    #endif

    do{
        //change
        // #if Choice
            mate1 = select1(&nremain, choices);//选择双亲
            mate2 = select1(&nremain, choices);//选择双亲
        // #else
        //     mate1 = ga_select(popsize, *sumfitness, oldpop);
        //     mate2 = ga_select(popsize, *sumfitness, oldpop);
        // #endif

        crossover(oldpop[mate1].chrom, oldpop[mate2].chrom,
                  &(newpop[j].chrom[0]), &(newpop[j+1].chrom[0]),
                  lchrom,ncross,nmutation,&jcross,pcross,pmutation,nvar);
        //生成的两个儿子，j 和 j+1
        //j
        for(k = 1; k <= nvar; k++)
        {
            kk = 2*k - 1;
            newpop[j].x[k] = chrom_map_x(newpop[j].chrom[k],rvar[kk],rvar[kk+1],lchrom);
        }
        //计算raw fitness
        newpop[j].objective = objfunc(newpop[j].x, nvar);
        newpop[j].parent1 = mate1;
        newpop[j].parent2 = mate2;

        newpop[j].xsite = jcross;
        //-----------
        //j+1
        for(k = 1; k <= nvar; k++)
        {
            kk = 2*k - 1;
            newpop[j+1].x[k] = chrom_map_x(newpop[j+1].chrom[k],rvar[kk],rvar[kk+1],lchrom);
        }
        //计算raw fitness
        newpop[j+1].objective = objfunc(newpop[j+1].x, nvar);
        newpop[j+1].parent1 = mate1;
        newpop[j+1].parent2 = mate2;

        newpop[j+1].xsite = jcross;
        //------------
        j += 2;

    }
    while(j < popsize);
}

// int ga_select(int popsize, double sumfitness, ind_t *pop)
// {
//     int j;
//     double partsum, randx;

//     partsum = 0.; j = 0;
//     randx = random1()*sumfitness;
//     do{
//         j += 1;
//         partsum += pop[j].fitness;
//     }
//     while(partsum <= randx && j != popsize);
//     return(j);
// }

void crossover(int *parent1,int *parent2,int *child1,int *child2,
               int lchrom,unsigned long *ncross,unsigned long *nmutation,int *jcross,
               double pcross,double pmutation, int nvar)
{
    register int k,kk;
    int j, lighted, test, rn;

    memcpy(child1, parent1, sizeof(int)*(nvar+1));
    memcpy(child2, parent2, sizeof(int)*(nvar+1));
    //pcross 是交叉的概率
    if(flip(pcross)  == 1)
    {
        //交叉执行，计算交叉位数jcross
        *jcross = rnd(1, nvar*lchrom-1);
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
        test = mutation(pmutation, nmutation);
        /*test = [0] no change , test = 1 bit changed kth bit is altered*/
        if(test == 1) bit_on_off(&child1[kk],k); /* mutation */

        test = mutation(pmutation, nmutation);
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
            test = mutation(pmutation, nmutation);
                   /*test = [0] no change , test = 1 bit changed jth bit is altered*/
            bit_on(&child1[kk],k);
            if(lighted == 0) bit_on_off(&child1[kk],k);
            if(test == 1)bit_on_off(&child1[kk],k);    /* mutate */

            lighted = bit_check(parent1[kk],k);        /*lighted = [1] if bit is on */
            test = mutation(pmutation, nmutation);
                   /*test = [0] no change , test = 1 bit changed jth bit is altered*/
            bit_on(&child2[kk],k);
            if(lighted == 0) bit_on_off(&child2[kk],k);
            if(test == 1)bit_on_off(&child2[kk],k);    /* mutate */
        }
    }
}

int round_(int j, int lchrom)
{

    if(fmod(j,lchrom) == 0)
    return(1);

    return(0);
}

int  rnd(int low,int high)
{
    int i;

    if(low >= high) i = low;
    else
    {
        i = (int) (random1()*(high-low+1) + low);
        if(i > high) i = high;
    }
    return(i);
}

int  mutation(double pmutation,unsigned long *nmutation)
{
    int mutate;

    mutate = flip(pmutation);
    if(mutate == 1)
    {
        *nmutation += 1;
        return(1);
    }
    else
        return(0);
}

void report(int gen,ind_t *oldpop,ind_t *newpop,
          int lchrom,double max,double avg,double min,double sumfitness,
          unsigned long nmutation,unsigned long ncross, int popsize,
          int nvar,max_t *max_ult)
{
    register int k;
    FILE *fpout;
    int j;
    double sum;

    if((fpout=fopen("genout.dat","a")) == (FILE *) NULL)
    {
        printf("cannot open genout.dat\n");
        exit(1);
    }

    fprintf(fpout,"Generation #%5d data (lchrom %5d)\n",gen,lchrom);
    fprintf(fpout,"No. parent1 parent2      x     fitness\n");
    fprintf(fpout,"----------------------------------\n");

    fprintf(fpout,"Old Generation of Generation No. %5d\n",gen);
    for(j = 1; j <= popsize; j++)
    {
        fprintf(fpout,"%5d %5d %5d",j,oldpop[j].parent1,oldpop[j].parent2);
        for(k=1; k <= nvar; k++)
            fprintf(fpout," %9.3e",oldpop[j].x[k]);
        fprintf(fpout,"    %9.3e\n",oldpop[j].objective);
    }

    fprintf(fpout,"New Generation of Generation No. %5d\n",gen);
    for(j = 1; j <= popsize; j++)
    {
        fprintf(fpout,"%5d %5d %5d",j,newpop[j].parent1,newpop[j].parent2);
        for(k=1; k <= nvar; k++)
            fprintf(fpout," %9.3e",newpop[j].x[k]);
        fprintf(fpout,"    %9.3e\n",newpop[j].objective);
    }

    fprintf(fpout,"New (%5d) generation's statistics\n",gen);

    fprintf(fpout," max=%10.5e avg=%10.5e min=%10.5e sum=%10.5e\n nmutation=%lu \
    ncross=%lu\n",max,avg,min,sumfitness,nmutation,ncross);

    fprintf(fpout,"\nMax of all generations\n");
    fprintf(fpout,"occurred in generation no.%5d \n",max_ult->gen);
    fprintf(fpout,"chromosome variables\n");
    for(k=1; k <= nvar; k++)
        fprintf(fpout," %5d",max_ult->chrom[k]);
    fprintf(fpout,"\n");
    fprintf(fpout,"design variables\n");
    for(k=1; k <= nvar; k++)
        fprintf(fpout," %9.3e",max_ult->x[k]);
    fprintf(fpout,"\n");
    fprintf(fpout,"raw fitness = %f\n",max_ult->objective);
    fclose(fpout);
}


void scalepop(int popsize, double max, double avg, double min,
              double *sumfitness,ind_t *pop)
{
    register int j;
    double a, b;

    prescale(max,avg,min,&a,&b);
    *sumfitness = 0.;

    for(j = 1; j <= popsize; j++)
    {
        //raw fitness --> fitness: f' = af + b;
        pop[j].fitness = a*pop[j].objective+b; 
        *sumfitness += pop[j].fitness;
    }
}

void prescale(double umax, double uavg, double umin, double *a, double *b)
{
    double delta;

    if(umin > (Fmultiple*uavg-umax)/(Fmultiple-1.)) /*Non-neg test*/
    {
        delta = umax-uavg;
        if(delta == 0.) delta = .00000001;
        *a = (Fmultiple-1.)*uavg/delta;
        *b = uavg*(umax-Fmultiple*uavg)/delta;
    }else{
        delta = uavg - umin;
        if(delta == 0.) delta = .00000001;
        *a = uavg/delta;
        *b = -umin*uavg/delta;
    }
}

void pre_select1(int popsize, double avg, ind_t *pop, int *choices)
{
    register int j, k;
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
            winner = flip(fraction[j]);
            if(winner == 1)
            {
                k++;
                choices[k]= j;
                (fraction[j])--;
            }
        }
    }
}

int select1(int *nremain,int *choices)
{
    int jpick, index;

    jpick = rnd(1, *nremain);

    index = choices[jpick];

    choices[jpick] = choices[*nremain];

    *nremain = *nremain - 1;

    return(index);
}

double objfunc(double *x, int nvar) //fitness 计算函数
{
    register int i,j;
    double res, result,penalty;

    result = 700 - (x[1]+pow(x[2],2)+x[2]*x[3]);
    penalty =1000 * pow(x[1]+x[2]+x[3],2) ;
    result -= penalty;

    if(fabs(result) <= 0. ) result = Tol2;

    return(result);
}
#endif