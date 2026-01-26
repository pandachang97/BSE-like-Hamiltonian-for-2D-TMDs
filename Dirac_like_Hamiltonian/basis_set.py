import scipy
import numpy as np
import math
from numba import njit
import json

# I want to include disorder in the future. I just define Hamiltonian and its Eigenvalue
#  as a class, then we can calculate Absorption and PL spectrum based on them.
with open('parameters.json') as IBS_inp:
    parameters = json.load(IBS_inp)

CT_inter_periodic = parameters["basis_set_options"]['CT_inter_periodic']
Nchrom = parameters['geometry_parameters']['Nchrom']
vibmax = parameters['geometry_parameters']['vibmax']

class IBS:
    def __init__( self ):
        self.Nchrom = Nchrom
        self.vibmax = vibmax
        self.CT_inter_periodic = CT_inter_periodic
        self.kcount = 0
        self.Index_single = np.zeros( ( self.Nchrom * ( self.vibmax + 1 )) , dtype = np.int64 )

        self.Index_double = np.zeros( ( self.Nchrom * ( self.vibmax + 1 ) * \
        self.Nchrom * self.vibmax ), dtype = np.int64 )

        self.Index_tripple = np.zeros( (  self.Nchrom * ( self.vibmax + 1 ) *  \
        self.Nchrom * self.vibmax * self.Nchrom * self.vibmax ) , dtype = np.int64 )

        self.Index_CT = np.zeros( ( self.Nchrom * ( self.vibmax + 1 ) * \
        self.Nchrom * ( self.vibmax + 1 ) ) , dtype = np.int64 )

        self.Index_CTv = np.zeros( ( self.Nchrom * ( self.vibmax + 1 ) *  \
        self.Nchrom * ( self.vibmax + 1 ) * self.Nchrom * self.vibmax) , dtype = np.int64 )

        self.Index_TP = np.zeros( ( self.Nchrom * ( self.vibmax + 1 ) * \
        self.Nchrom * ( self.vibmax + 1 ) ) , dtype=np.int64)  #Triplet pair state

        self.Index_TPv = np.zeros( ( self.Nchrom * ( self.vibmax + 1 ) *  \
        self.Nchrom * ( self.vibmax + 1 ) * self.Nchrom * self.vibmax ) , dtype=np.int64) #Triplet pair state with vibrations

#define a function to calculate the running label of 1 particle singlet array
    #@njit( fastmath=True)
    def order_1p( self, i, i1 ):
        num_1p = ( i ) * ( self.vibmax + 1 ) + i1 
        return num_1p
#Indexing 1p basis set below:
    #@njit(fastmath=True)
    def arr_1p(self):
        for i in range( 0 , self.Nchrom  ):
            for i1 in range( 0 , self.vibmax + 1 ):
                    a = self.order_1p( i , i1  )
                    self.Index_single[ a ] = self.kcount 
                    self.kcount = self.kcount + 1
        return self.Index_single
#define a function to calculate the running label of 2 particle singlet array
    #@njit(fastmath=True)
    def order_2p( self, i, i1, j, j1 ):
        num_2p =  ( i ) * ( self.vibmax + 1 ) *self.Nchrom * self.vibmax + \
               ( i1 ) * self.Nchrom * self.vibmax + \
               ( j ) * self.vibmax + \
               j1
        return num_2p 
#Indexing 2p basis set below:
    #@njit(fastmath=True)
    def arr_2p( self ):
        for i in range( 0 , self.Nchrom ):
            for i1 in range( 0 , self.vibmax + 1 ):
                for j in range( 0 , self.Nchrom ):
                    for j1 in range( 0 , self.vibmax ):
                        if  i == j:
                             continue
                        else:
                            if i1 + j1 + 1 > self.vibmax:
                                continue
                            else:  
                                a = self.order_2p( i, i1, j, j1 )
                                self.Index_double[ a ] = self.kcount
                                self.kcount = self.kcount + 1
        return self.Index_double
#define a function to calculate the running label of 3 particle singlet array
    #@njit(fastmath=True)
    def order_3p( self, i, i1,j, j1, k, k1 ):
        num_3p =( i - 1 ) * ( self.vibmax + 1 ) * self.Nchrom * self.vibmax * self.Nchrom * self.vibmax + \
               ( i1 - 1 ) * self.Nchrom * self.vibmax * self.Nchrom * self.vibmax + \
               ( j - 1 ) * self.vibmax * self.Nchrom * self.vibmax + \
               ( j1 - 1 ) * self.Nchrom * self.vibmax + \
               ( k - 1 ) * self.vibmax + \
               k1
        return num_3p
#Indexing 3p basis set below:
    #@njit(fastmath=True)
    def arr_3p( self ):
        for i in range( 0 , self.Nchrom ):
            for i1 in range( 0 , self.vibmax + 1 ):
                for j in range( 0 , self.Nchrom ):
                    for j1 in range( 0 , self.vibmax ):
                       for k in range( 0 , self.Nchrom ):
                            for k1 in range( 0 , self.vibmax ):
                                if ( i == j or  j >=  k or  k == i ):
                                    continue
                                else:
                                    if i1 + j1 + 1 + k1 + 1 > self.vibmax:
                                        continue
                                    else:
                                        if ( ( abs( i - j ) == 1 ) or abs( i - k ) > 1 and abs( i - k ) != self.Nchrom - 1 ):
                                            a = self.order_3p(  i, i1, j, j1, k, k1 )
                                            self.Index_tripple[ a ] = self.kcount
                                            self.kcount = self.kcount + 1
        return self.Index_tripple
#define a function to calculate the running label of CT array
    #@njit(fastmath=True)
    def order_CT( self, i, i1, j, j1 ):
        num_CT = ( i ) * ( self.vibmax + 1 ) * self.Nchrom * ( self.vibmax + 1 ) + \
                 ( i1 ) * self.Nchrom * ( self.vibmax + 1 ) + \
                 ( j  ) * ( self.vibmax + 1 ) + \
                j1 
        return num_CT
#Indexing CT basis set below:
    #@njit(fastmath=True)
    def arr_CT( self ):
        for i in range( 0 , self.Nchrom ):
            for i1 in range( 0 , self.vibmax + 1 ):
                for j in range( 0 , self.Nchrom ):
                    for j1 in range( 0 , self.vibmax + 1 ):
                        if i == j:
                            continue
                        else:
                            if( ( ( abs( i - j ) == 1 ) ) or  \
          ( self.Nchrom >= 3 and abs( i - j ) == self.Nchrom - 1 and self.CT_inter_periodic ) ):
                                if ( i1 + j1 > self.vibmax ):
                                    continue
                                else:
                                    a = self.order_CT( i, i1, j, j1 )
                                    self.Index_CT[ a ] = self.kcount
                                    self.kcount = self.kcount + 1
        return self.Index_CT 
#define a function to calculate the running label of CTv array
    #@njit(fastmath=True)
    def order_CTv( self, i, i1, j, j1, k, k1 ):
        num_CTv =( i - 1 ) * ( self.vibmax + 1 ) * self.Nchrom * ( self.vibmax + 1 ) * self.Nchrom * self.vibmax + \
                 ( i1 - 1 ) * self.Nchrom * ( self.vibmax + 1 ) * self.Nchrom * self.vibmax + \
                 ( j - 1 ) * ( self.vibmax + 1 ) *  self.Nchrom * self.vibmax + \
                 ( j1 - 1 ) * self.Nchrom * self.vibmax + \
                 ( k - 1 ) * self.vibmax + \
                 k1 
        return num_CTv
#Indexing CTv basis set below:
    #@njit(fastmath=True)
    def arr_CTv( self ):
        for i in range( 0 , self.Nchrom ):
            for i1 in range( 0 , self.vibmax + 1 ):
                for j in range( 0 , self.Nchrom ):
                    for j1 in range( 0 , self.vibmax + 1 ):
                        for k in range( 0 , self.Nchrom ):
                            for k1 in range( 0 , self.vibmax ):
                                if (  i == j or j == k or i  == k ) :
                                    continue
                                else:
                                    if ( ( abs( i - j) == 1  ) or \
                                    ( self.Nchrom >= 3 and abs( i - j) == self.Nchrom - 1 and self.CT_inter_periodic ) ):
                                        if ( i1 + j1 + k1 + 1 > self.vibmax ):
                                            continue
                                        else:
                                            if ( abs( i - k ) == 1 ) or (abs( j - k ) == 1 ): 
                                                a = self.order_CTv( i, i1, j, j1, k, k1 )
                                                #print(f"a is {a}")
                                                self.Index_CTv[ a ] = self.kcount
                                                self.kcount = self.kcount + 1
        return self.Index_CTv
#define a function to calculate the running label of Triple Pair array    
    #@njit(fastmath=True)
    def order_TP( self, i, i1, j, j1 ):
        num_TP = ( i ) * ( self.vibmax + 1 ) * self.Nchrom * ( self.vibmax + 1 ) + \
                 ( i1 ) * self.Nchrom * ( self.vibmax + 1 ) + \
                 ( j  ) * ( self.vibmax + 1 ) + \
                j1 
        return num_TP
#Indexing Triplet Pair basis set below:
    #@njit(fastmath=True)
    def arr_TP( self ):
        for i in range( 0 , self.Nchrom ):
            for i1 in range( 0 , self.vibmax + 1 ):
                for j in range( 0 , self.Nchrom ):
                    for j1 in range( 0 , self.vibmax + 1 ):
                        if i >= j:
                            continue
                        else:
                            if( ( ( abs( i - j ) == 1 ) ) or  \
                            ( self.Nchrom >= 3 and abs( i - j ) == self.Nchrom - 1 and self.CT_inter_periodic ) ):
                                if ( i1 + j1 > self.vibmax ):
                                    continue
                                else:
                                    a = self.order_TP( i, i1, j, j1 )
                                    self.Index_TP[ a ] = self.kcount
                                    self.kcount = self.kcount + 1
        return self.Index_TP
#define a function to calculate the running label of Triplet Pari with virbration array
    #@njit(fastmath=True)
    def order_TPv( self, i, i1, j, j1, k, k1 ):
        num_TPv =( i - 1 ) * ( self.vibmax + 1 ) * self.Nchrom * ( self.vibmax + 1 ) * self.Nchrom * self.vibmax + \
                 ( i1 - 1 ) * self.Nchrom * ( self.vibmax + 1 ) * self.Nchrom * self.vibmax + \
                 ( j - 1 ) * ( self.vibmax + 1 ) *  self.Nchrom * self.vibmax + \
                 ( j1 - 1 ) * self.Nchrom * self.vibmax + \
                 ( k - 1 ) * self.vibmax + \
                 k1 
        return num_TPv
#Indexing Triplet Pari with virbration basis set below:
    #@njit(fastmath=True)
    def arr_TPv( self ):
        for i in range( 0 , self.Nchrom ):
            for i1 in range( 0 , self.vibmax + 1 ):
                for j in range( 0 , self.Nchrom ):
                    for j1 in range( 0 , self.vibmax + 1 ):
                        for k in range( 0 , self.Nchrom ):
                            for k1 in range( 0 , self.vibmax ):
                                if (  i >= j or j == k or i  == k ) :
                                    continue
                                else:
                                    if ( ( abs( i - j) == 1  ) or \
                                    ( self.Nchrom >= 3 and abs( i - j) == self.Nchrom - 1 and self.CT_inter_periodic ) ):
                                        if ( i1 + j1 + k1 + 1 > self.vibmax ):
                                            continue
                                        else:
                                            if ( abs( i - k ) == 1 ) or (abs( j - k ) == 1 ): 
                                                a = self.order_TPv( i, i1, j, j1, k, k1 )
                                                #print(f"a is {a}")
                                                self.Index_TPv[ a ] = self.kcount
                                                self.kcount = self.kcount + 1
        return self.Index_TPv   
    
##Finally, we need to figure out the total dimenston of the basis set.
    def tot_dim(self):
        return self.kcount