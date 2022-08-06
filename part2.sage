import time
import sys
import os

def id_rect(m,n,Q):
    mat = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(min(m,n)):
        mat[i][i]=Q
    return mat

def cf_approx(mat,bound):
    # calculates a rational approximation of a matrix
    # using continued fraction approximation
    B0 = matrix(QQ,mat.nrows(),mat.ncols())
    for i in range(mat.nrows()):
        for j in range(mat.ncols()):
            cc = continued_fraction(mat[i,j])
            approx = 0#cc.convergent(0)
            for k in range(len(cc)):
                if cc.convergent(k).denominator()>bound:
                    break
                approx = cc.convergent(k)
            B0[i,j] = approx
    return B0

def ppm(mat):
    for row in mat:
        for ent in row:
            if abs(ent) < 1e-20:
                print('0',end=' ')
            elif ent.denominator() > 1e10:
                print(ent.n(digits=4),end=' ')
            else:
                print(ent,end=' ')
        print()

def AlgorithmA2(ys,l,N,Q,R,T,verbose):
    # Defines the second part of Algorithm A

    # input parameters:
    #   ys: an array of simulated measurements 
    #       note: the measurement y must be taken using Fnorms
    #       from part 1, prior to executing part 2
    #   l: dimension of the hidden lattice H
    #   Q: resolution of the hypercube
    #   R: a bound on the denominator of the LLL output
    #   T: the flattening scale of the dual lattice
    #   verbose: Boolean, whether to print logging and debugging information

    # output:
    #   A1: a matrix which hopefully generates the original lattice H

    ysamples = []
    m = len(ys) # number of samples
    k = len(ys[0])
    for i,y in enumerate(ys):
        y2 = [j/Q for j in y]
        for ix in range(m):
            if ix==i:
                y2.append(1/T)
            else:
                y2.append(0)
        ysamples.append(y2)

    basis_vectors = id_rect(k,k+m,1)
    for y in ysamples:
        basis_vectors.append(y)
    
    basis_vectors = Matrix(QQ,basis_vectors)

    # if verbose:
    #     print('The original basis vectors are:')
    #     print(basis_vectors.n(digits=5))
    #     print()
        
    #apply the LLL algorithm, as implemented in SageMath, to the basis vectors
    lllbasis = basis_vectors.LLL()
    if verbose:
        print('The LLL reduced basis vectors are:')
        for bvec in lllbasis:
            print('%s - %s' %(str(bvec.norm().n(digits=5)),bvec.n(digits=5)))
        print()
        
    #calculate B1, the submatrix of vectors with small norm
    Btemp = []
    ell = lllbasis.nrows()
            
    #this is an ordered search to match the ordered basis
    for i in lllbasis:
        if i.norm() > 1/R:
            break
        ell -= 1
        Btemp.append(i)

    Btemp = Matrix(Btemp).T
    B1 = Matrix(Btemp)

    if ell != l:
        if verbose:
            print('Fatal error: %i vectors were deleted from B. Expected: %i.\n' % (ell,l))
        return None
    
    if verbose:
        print('The last %i basis vector(s) were deleted.' % (ell))
        # print('The last %i basis vector(s) were deleted, giving B1 = ' % (ell))
        # print(B1.n(digits=5))
        print()
    
    TT = Matrix(RR,B1)

    #calculate B2
    B2 = Matrix(Btemp)
    # if verbose:
    #     print('Creating B2 from B1...')
        
    while B2.nrows() != B2.ncols():
        maxDet = -1
        iTemp = -1
        for i in range(B2.nrows()):
            B2j = B2.delete_rows([i])
            tempDet = abs((B2j.T*B2j).determinant())
            if tempDet > maxDet:
                maxDet = tempDet
                iTemp = i
                
        B2 = B2.delete_rows([iTemp])
        if verbose:
            print('We delete row %i.' % (iTemp))
        #     print('B2 now equals:')
        #     print(B2.n(digits=sf))
        #     print()
    
    if verbose:
        print()
        print('Ultimately, B2 = ')
        ppm(B2)#print(B2)
        print()
    
    try:
        B3 = B1*(B2^(-1))
        if verbose:
            print('B3 = B1 x B2^(-1) = ')
            ppm(B3)#print(B3)
            print()
    except:
        print('Nonsingularity, or invalid multiplication, found!')
        print(B2)
        return None

    B0 = cf_approx(B3,round((l**(l/2))*(N**l)))

    if verbose:
        print('B3 is rationalized to give B0 = ')
        print(B0)
        print()

    return B0

def dual_lattice_sample(Hmatrix,Q,precision):
    #smith_form of a matrix A returns matrices D,U,V
    #such that UAV = D
    D,U,_ = Hmatrix.smith_form()

    Dnp = [list(i) for i in D]
    ix = 0
    for row in Dnp:
        if sum(row) == 0:
            break
        ix += 1    
    W = []
    for i in range(ix):
        temp = Dnp[i][i]
        W.append(randint(0,temp-1))

    for _ in range(len(Dnp)-ix):
        tracebound = round(sqrt((U.T*U).trace()))
        val = Q*tracebound*precision
        bound = round(val)
        wval = randint(0,bound-1)/bound
        W.append(wval)

    WU = (Matrix(QQ,W)*U).list()
    modmatrix = [i-round(i) for i in WU]
    y = [round(Q*i) for i in modmatrix]
    return y

def explore(its,m,N,k,l,R,T,Q):
    success = 0
    failure = 0
    nothing = 0
    misses = []
    for _ in range(its):
        H = random_matrix(ZZ,k,l,x=-N,y=N+1)
        Hech = Matrix(QQ,H.T).echelon_form()
        ysamples = []
        for _ in range(m):
            currSample = dual_lattice_sample(H,Q,10)
            ysamples.append(currSample)
        A1 = AlgorithmA2(ysamples,l,N,Q,R,T,False)
        if A1:
            #https://doc.sagemath.org/html/en/reference/matrices/sage/matrix/matrix2.html
            A1ech = Matrix(QQ,A1.kernel(algorithm='generic').basis())[:,:-m]
            if A1ech == Hech:
                success += 1
            else:
                failure += 1
                misses.append(A1)  
        else:
            nothing += 1
    
    return success,failure,nothing,misses

sample = False
if sample:
    N = 5*10**1
    print('N = %.2E.\n' %(N))

    k = 4
    l = 2
    H = random_matrix(ZZ,k,l,x=-N,y=N+1)
    
    print('H = ')
    print(H)
    print()
    
    Hech = Matrix(QQ,H.T).echelon_form()
    print('H has row space')
    print(Hech)
    print()

    m = 1

    R = 2*N
    print('R = %.2E. 1/R = %.4E.' % (R,1/R))

    T = round(1000*(R*N**(k/m)))
    print('T = %.2E. 1/T = %.4E.' % (T,1/T))

    Q = 20*R*T
    print('Q = %.2E.\n' % (Q))

    ysamples = []
    for _ in range(m):
        ysamples.append(dual_lattice_sample(H,Q,10))
    # for ys in ysamples:
    #     print(["%.2E"%(i) for i in ys])
    print()

    verbose = False

    A1 = AlgorithmA2(ysamples,l,N,Q,R,T,verbose)

    A1ech = None
    if A1:
        if not verbose:
            print('The predicted generator matrix is:')
            print(A1)
            print()
        A1ech = Matrix(QQ,A1.kernel(algorithm='generic').basis())[:,:-m]
        print('The predicted kernel is:')
        print(A1ech)
        print()
    else:
        print('Error: No suitable generator matrix was found.\n')

    if A1ech==Hech:
        print('The kernels match and the test instance has succeeded.')
    else:
        print('The test instance has failed.')
    print()
    t1 = time.time()
    su,fa,no,misses = explore(20,m,N,k,l,R,T,Q)
    print(time.time()-t1)
    print('Repeated tests scored %i - %i - %i.\n' % (su,fa,no))
    sys.exit()

def searchAll(k,l,m,N,numTests,threshold,verbose):
    totalTests = 0
    done = False
    t1 = time.time()
    limit = 7200
    alpha = sqrt(k)/1.6
    resetBeta = True

    while not done:
        if time.time() - t1 > limit:
            print('Timeout: %i seconds elapsed.' % (limit))
            return 0,0,0
        R = round(N*alpha)
        if resetBeta:
            beta = 1

        beginGammaSearch = False
        while beta < 10**40:
            beta *= 2.5
            T = round(beta*(R*(N**(k/m))))
            Q = 2*R*T*(N**k)
            su,fa,no,_ = explore(numTests,m,N,k,l,R,T,Q)
            totalTests += 1
            if verbose:
                print("(%i,%i,%i): N = %.0E, a = %.2f, b = %.2E. Q = %.1E. Score: %i-%i-%i."
                    % (k,l,m,N,alpha,beta,Q,su,fa,no))
            if su >= numTests*threshold:
                beginGammaSearch = True
                break
        
        resetBeta = True
        if not beginGammaSearch:
            alpha += 0.2

        if beginGammaSearch:
            if verbose:
                print('R and T found. Searching for Q...')
            gamma = 1
            gammaFound = False
            while gamma < 50:
                gamma *= 1.2
                Q = round(gamma*2*R*T)
                su,fa,no,_ = explore(numTests,m,N,k,l,R,T,Q)
                totalTests += 1
                if verbose:
                    print("g = %.3f. Score: %i - %i - %i." % (gamma,su,fa,no))
                if su >= numTests*threshold:
                    gammaFound = True
                    break
            if gammaFound:
                done = True
            else:
                resetBeta = False
                if verbose:
                    print('Q not found. Insufficient R and T. Retrying...')

    if verbose:
        print('Done!')

    return alpha,beta,gamma,Q,totalTests

numTests = 20
threshold = .666
verbose = True

if not os.path.exists('logs'):
    os.makedirs('logs')

with open('logs\%s.csv'%(time.ctime()).replace(':','-'),'w') as log:
    numIts = 0
    t1 = time.time()
    log.write('k,l,m,N,alpha,beta,gamma,Q,time\n')
    ks = [4,8]
    for k in ks:
        ells = set([round(k/2)])#k-1
        for l in ells:
            Ns = [10**i for i in range(1,10)]#,10000,1000000]
            for N in Ns:
                ms = [1,4]
                for m in ms:
                    t0 = time.time()
                    alpha,beta,gamma,Q,tTs = searchAll(k,l,m,N,numTests,threshold,verbose)
                    numIts += tTs
                    logline = '%i,%i,%i,%.0E,%.3f,%.2E,%.3f,%.3E,%.1f\n' % (k,l,m,N,alpha,beta,gamma,Q,time.time()-t0)
                    log.write(logline)
    print('%i tests done in %.3f seconds.' % (numIts*numTests,time.time()-t1))