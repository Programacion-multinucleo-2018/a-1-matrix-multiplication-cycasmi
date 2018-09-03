/*
 * File:   main.cpp
 * Author: Cynthia Castillo
 * Student ID: A01374530
 *
 * Created on September 1st, 2018, 08:13 PM
 */

#include "Matrix.h"
#include <chrono>
#include <iomanip>
using namespace std;

void matrixMult(Matrix *A, Matrix *B, Matrix *C)
{
     if (A == NULL || B == NULL)
     {
        printf("Multiplication failed.\n One or more of the matrices are Empty\n");        return;
     }

    if (A->cols != B->rows)
    {
        printf("Incompatible matrix dimensions\n");
        return;
    }

    int  shared_dim = 1;
    for (int i = 1; i <= C->rows; i++)
    {
        for (int j = 1; j <= C->cols; j++)
        {
            ELEM(C, i, j) = 0;
            for (shared_dim = 1; shared_dim <= A->cols; shared_dim++)
            {
                //dot product
                ELEM(C, i, j) += ELEM(A, i, shared_dim) * ELEM(B, shared_dim, j);
            }
        }
    }

    return;
    }

void ompMatrixMult(Matrix *A, Matrix *B, Matrix *C)
{
     if (A == NULL || B == NULL)
     {
        printf("Multiplication failed.\n One or more of the matrices are Empty\n");
        return;
     }

    if (A->cols != B->rows)
    {
        printf("Incompatible matrix dimensions\n");
        return;
    }


    int  i, j, shared_dim = 1;
    #pragma omp parallel for private(i, j, shared_dim) shared(A, B, C) collapse(2)
    for (i = 1; i <= C->rows; i++)
    {
        for (j = 1; j <= C->cols; j++)
        {
            ELEM(C, i, j) = 0;
            for (shared_dim = 1; shared_dim <= A->cols; shared_dim++)
            {
                //dot product
                ELEM(C, i, j) += ELEM(A, i, shared_dim) * ELEM(B, shared_dim, j);
            }
        }
    }

    return;
}

int main(int argc, char** argv)
{
    Matrix *A, *B, *C;

    int x = 0, y = 0;

    if(argc < 2)
        {
        x = y = 1000;
    }
    else
    {
        x = y = stoi(argv[1]);
    }


    A = new Matrix(x, y);
    A->randFill();
    //A->print();

    cout << endl;

    B = new Matrix(x, y);
    B->randFill();
    //B->print();

    C = new Matrix(x, y);

    auto start = chrono::high_resolution_clock::now();
    matrixMult(A, B, C);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end - start;
    printf("time seq (ms): %f\n", duration_ms.count());
    //C->print();
    //cout << endl;


    start = chrono::high_resolution_clock::now();
    ompMatrixMult(A, B, C);
    end = chrono::high_resolution_clock::now();
    duration_ms = end - start;
    printf("time omp (ms): %f\n", duration_ms.count());
    //C->print();

    delete A, B, C;

    return 0;
 }
