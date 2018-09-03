
/*
 * File:   Matrix.h
 * Author: Cynthia Castillo
 * Student ID: A01374530
 *
 * Created on September 1st, 2018, 08:13 PM
 */

#ifndef MATRIX_H
#define MATRIX_H

/*Macro definition for accessing matrix elements
 * in Matrix format
 */
#define ELEM(m, row, col) \
  m->data[(col-1) * m->rows + (row-1)]

#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;

class Matrix
{
    public:
        int rows, cols;
        int* data;

        Matrix()
        {
            rows = 0;
            cols = 0;
            data = NULL;
        }

        Matrix(int _rows, int _cols)
        {
            //Invalid dimensions for the matrix. Exits the program
            if (_rows <= 0 || _cols <= 0)
            {
                            cout << "Invalid Matrix dimensions" << endl;
                exit(1);
            }

            // set matrix dimensions
            rows = _rows;
            cols = _cols;
            // allocate a int array of length rows * cols
            data = (int *) malloc(rows*cols*sizeof(int));
        }

        void randFill()
        {
            srand (time(NULL));
            for (int i = 1; i <= rows; i++)
            {
                for (int j = 1; j <= cols; j++)
                {
                    ELEM(this, i, j) = rand() % 10 + 1;
                }
            }
        }

        void print()
        {
            if (this == NULL || data == NULL)
                cout << "Nothing to print. Matrix is empty\n" << endl;
            else
            {
                for (int i = 1; i <= rows; i++)
                {
                  for (int j = 1; j <= cols; j++)
                      cout << ELEM(this, i, j) << " ";

                  //newline for each row
                  cout << endl;
                }
            }
        }
};
