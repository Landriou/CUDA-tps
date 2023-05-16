#include <iostream>
#include <sys/time.h>
using namespace std;

int main() {
    cout<<  "inicie" << endl;
    struct timeval t1, t2;

    gettimeofday(&t1, 0);
    int s = 3;
    int column = s;
    int row = s;
    int a[s][s], transpose[s][s];


   int count = 0;
   for (int i = 0; i < row; ++i) {
      for (int j = 0; j < column; ++j) {
          a[i][j] = count;
          count++;
      }
   }

   // Printing the a matrix
   cout << "\nEntered Matrix: " << endl;
   for (int i = 0; i < row; ++i) {
      for (int j = 0; j < column; ++j) {
         cout << " " << a[i][j];
         if (j == column - 1)
            cout << endl << endl;
      }
   }

   for (int i = 0; i < row; ++i)
      for (int j = 0; j < column; ++j) {
         transpose[j][i] = a[i][j];
      }

   cout << "\nTranspose of Matrix: " << endl;
   for (int i = 0; i < column; ++i)
      for (int j = 0; j < row; ++j) {
         cout << " " << transpose[i][j];
         if (j == row - 1)
            cout << endl << endl;
      }
        gettimeofday(&t2, 0);
double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;
   cout<< "el tiempo que tardo fue "<< time << endl; 

  return 0;
}