#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void extract_pulses_in_blocks(const int n, const double period_ms, const double dm, const double length,
                              char* maskfile, char* rficleaned_file, int npulses_nfiles[]);

int main(int argc, char **argv)
{

    printf("--------------------------\n");
    printf("In extract_single_pulses.C\n");
    printf("--------------------------\n");

    if (argc != 6)
    {
        printf("Must supply exacly five arguments\n");
        printf("Number of arguments supplied = %d\n", argc);
        exit(-1);
    }

#ifdef _OPENMP
    printf("Using OpenMP!\n");
    omp_set_dynamic(0);     // Explicitly disable dynamic teams
    omp_set_num_threads(32); // Use 4 threads for all consecutive parallel regions
    const int total_threads = omp_get_max_threads();
#else
    const int total_threads = 1;
#endif
    printf("Running waterfaller with %d threads in total.\n", total_threads);

    const double period_ms = atof(argv[1]); // initial period, in milliseconds.
    const double dm = atof(argv[2]);        // dispersion measurement.
    const double length = atof(argv[3]);    // length of the observation in seconds.
    char* maskfile = argv[4];               // name of the mask.
    char* rficleaned_file = argv[5];        // name of the fil file that has been cleaned using RFIFind.

    printf("period_ms = %20.16e\n", period_ms);
    printf("dm = %20.16e\n", dm);
    printf("length = %20.16e\n", length);
    printf("maskfile = %s\n", maskfile);
    printf("rficleaned_file = %s\n", rficleaned_file);

    int npulses_nfiles[2]; // npulses_nfiles[0] : number of single pulses that have already been extracted at any single iteration
                           // npulses_nfiles[1] : number of files that have been created so far

    npulses_nfiles[0] = 0;
    npulses_nfiles[1] = 0;

    int sizes[4] = {100, 10, 1};

    for (int k = 0; k < 3; k++)
    {
        int n = sizes[k]; // number of single pulses in one block

        extract_pulses_in_blocks(n, period_ms, dm, length, maskfile, rficleaned_file, npulses_nfiles);
    }

//  Write results to output file
    FILE *fp = fopen("extraction_output.txt", "w+");
    if (fp == NULL)
    {
       printf("Could not open file");
       return 0;
    }
    fprintf(fp, "npulses %d\n", npulses_nfiles[0]);
    fprintf(fp, "nfiles %d\n", npulses_nfiles[1]);
    fclose(fp);
}

void extract_pulses_in_blocks(const int n, const double period_ms, const double dm, const double length,
                              char* maskfile, char* rficleaned_file, int npulses_nfiles[])
{
    int j = 1000 / n;
    int n_pulses_o_n = (length * j) / period_ms;     // number of blocks of n single pulses in the whole observation
    int prev_extracted = npulses_nfiles[0];          // number of single pulses previously extracted by the other block size

    int blocks_to_extract = (n_pulses_o_n * n - prev_extracted) / n; // number of blocks of n single pulses we still have to extract

//pragma omp parallel for reduction(+ : npulses_nfiles[:2])
    for (int i = 1; i <= blocks_to_extract; i++)
    {
        printf("Waterfaller dumping %d x %d of %d x %d \n", i, n, n_pulses_o_n, n);

        int idx = (i - 1) + (prev_extracted / n);    // block index
        double t_beg = (idx * period_ms) / j;        // time at the beginning of each block
        double deltaT = period_ms / j;               // time size of each block of n single pulses
        npulses_nfiles[0] += n;                      // we increase the number of pulses that have been extracted

	    // const int thread_id = omp_get_thread_num();
        const int thread_id = 0;

        char command[300];
	    printf("/home/jovyan/work/test/waterfaller_puma_numba -T %20.16e -t %20.16e --mask --maskfile %s -d %20.16e %s %d %d\n", t_beg, deltaT, maskfile, dm, rficleaned_file, thread_id, npulses_nfiles[1]);
        snprintf(command, 300, "/home/jovyan/work/test/waterfaller_puma_numba -T %20.16e -t %20.16e --mask --maskfile %s -d %20.16e %s %d %d\n", t_beg, deltaT, maskfile, dm, rficleaned_file, thread_id, npulses_nfiles[1]);
        int systemRet = system(command);
        if(systemRet == -1){
            printf("Waterfaller has failed\n"); // The system method failed
            exit(-1);
        }
        system(command); // execute waterfaller

        char new_times[40];
        char new_original[40];
        snprintf(new_times, 40, "times_%d.csv", npulses_nfiles[1]);
        snprintf(new_original, 40, "original_%d.csv", npulses_nfiles[1]);
        rename("times.csv", new_times);
        rename("original.csv", new_original);

        npulses_nfiles[1] += 1;
    }

}
