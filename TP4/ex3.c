#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void task_light(int N) {
    double x = 0.0;
    for (int i = 0; i < N; i++) {
        x += sin(i * 0.001);
    }
}

void task_moderate(int N) {
    double x = 0.0;
    for (int i = 0; i < 5*N; i++) {
        x += sqrt(i * 0.5) * cos(i * 0.001);
    }
}

void task_heavy(int N) {
    double x = 0.0;
    for (int i = 0; i < 20*N; i++) {
        x += sqrt(i * 0.5) * cos(i * 0.001) * sin(i * 0.0001);
    }
}

int main() {
    int N = 1000000;
    double start, end;
    
    printf("=== Load Balancing with OpenMP ===\n");
    printf("N = %d, Threads = %d\n\n", N, omp_get_max_threads());
    
    // Sequential baseline
    start = omp_get_wtime();
    task_light(N);
    task_moderate(N);
    task_heavy(N);
    end = omp_get_wtime();
    double seq_time = end - start;
    printf("Sequential: %.6f seconds\n\n", seq_time);
    
    // ============================================
    // Strategy 1: Basic sections (unbalanced)
    // ============================================
    printf("--- Strategy 1: Basic Parallel Sections (Unbalanced) ---\n");
    start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            printf("Thread %d: Task A (light)\n", omp_get_thread_num());
            task_light(N);
        }
        
        #pragma omp section
        {
            printf("Thread %d: Task B (moderate)\n", omp_get_thread_num());
            task_moderate(N);
        }
        
        #pragma omp section
        {
            printf("Thread %d: Task C (heavy)\n", omp_get_thread_num());
            task_heavy(N);
        }
    }
    end = omp_get_wtime();
    printf("Time: %.6f s, Speedup: %.2fx\n\n", 
           end - start, seq_time / (end - start));
    
    // ============================================
    // Strategy 2: Balanced sections (combine light tasks)
    // ============================================
    printf("--- Strategy 2: Balanced Sections (Combine A+B) ---\n");
    start = omp_get_wtime();
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            printf("Thread %d: Task A + B (light + moderate)\n", omp_get_thread_num());
            task_light(N);
            task_moderate(N);
        }
        
        #pragma omp section
        {
            printf("Thread %d: Task C (heavy)\n", omp_get_thread_num());
            task_heavy(N);
        }
    }
    end = omp_get_wtime();
    printf("Time: %.6f s, Speedup: %.2fx\n\n", 
           end - start, seq_time / (end - start));
    
    // ============================================
    // Strategy 3: Dynamic task scheduling
    // ============================================
    printf("--- Strategy 3: Dynamic Task Scheduling ---\n");
    start = omp_get_wtime();
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                printf("Thread %d: Task A assigned\n", omp_get_thread_num());
                task_light(N);
            }
            
            #pragma omp task
            {
                printf("Thread %d: Task B assigned\n", omp_get_thread_num());
                task_moderate(N);
            }
            
            #pragma omp task
            {
                printf("Thread %d: Task C assigned\n", omp_get_thread_num());
                task_heavy(N);
            }
        }
    }
    end = omp_get_wtime();
    printf("Time: %.6f s, Speedup: %.2fx\n\n", 
           end - start, seq_time / (end - start));
    
    // ============================================
    // Strategy 4: Priority-based scheduling
    // ============================================
    printf("--- Strategy 4: Priority-based Scheduling ---\n");
    start = omp_get_wtime();
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task priority(100)
            {
                printf("Thread %d: Task C (HIGH priority)\n", omp_get_thread_num());
                task_heavy(N);
            }
            
            #pragma omp task priority(50)
            {
                printf("Thread %d: Task B (MEDIUM priority)\n", omp_get_thread_num());
                task_moderate(N);
            }
            
            #pragma omp task priority(10)
            {
                printf("Thread %d: Task A (LOW priority)\n", omp_get_thread_num());
                task_light(N);
            }
        }
    }
    end = omp_get_wtime();
    printf("Time: %.6f s, Speedup: %.2fx\n\n", 
           end - start, seq_time / (end - start));
    
    // ============================================
    // Strategy 5: Parallel loop inside heavy task
    // ============================================
    printf("--- Strategy 5: Parallelize Heavy Task Internally ---\n");
    start = omp_get_wtime();
    
    task_light(N);
    task_moderate(N);
    
    double x = 0.0;
    #pragma omp parallel for reduction(+:x) schedule(dynamic, 1000)
    for (int i = 0; i < 20*N; i++) {
        x += sqrt(i * 0.5) * cos(i * 0.001) * sin(i * 0.0001);
    }
    
    end = omp_get_wtime();
    printf("Time: %.6f s, Speedup: %.2fx\n\n", 
           end - start, seq_time / (end - start));
    
    return 0;
}
