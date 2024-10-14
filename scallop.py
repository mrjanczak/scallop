'''
Scallop tool to construct 8 types of scallop, 
then generate mesh as an input to Siesta Arbitrary Scallop

author: Michal1.Janczak@ge.com
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

def ang(x):
    return np.array([np.sin(x*np.pi/180), np.cos(x*np.pi/180)])

class LARC:
    def __init__(self, P1, P2, PC, RAD, C):
        self.P1 = P1
        self.P2 = P2
        self.PC = PC
        self.RAD = RAD
        self.C = C

class L:
    def __init__(self, P1, P2):
        self.P1 = P1
        self.P2 = P2

def scallop(type = 4, N = 5, SANG = 30, D = 1,
    Ri = 4, Rp = 6, Ro = 8, Rext = 10, Rfil = .5,
    HW = .8, HH = 1, T = .5,
    R1 = 1, R2 = .5, R3 = 1,
    min_dR = .001):

    contour = []
    teta= 360/N/2
    print(f'teta: {teta}')

    points_no = 16
    P = np.zeros((points_no,2)) # start and end points of lines and arcs
    C = np.zeros((points_no,2)) # center of arc
    PC = np.zeros((points_no,2)) # midpoints of arcs

    #----------------------------
    P[0] = (Rext)*ang(0)
    P[9] = (Ri)*ang(0)

    if HW and HH and T and Rp and R1:
        P[1] = (Rp)*ang(0) + np.array([0, (HH+T)/2])
        P[4] = (Rp)*ang(0) + np.array([HW, T/2])

        dR12 = R1-R2
        dR23 = R2-R3

        if abs(dR12) < min_dR:
            dR12 = None
        if abs(dR23) < min_dR:
            dR23 = None

        # R2=R3=0
        if not dR12 and not dR23:
            alfa = 90
            beta = 0
            gamma = 0

        # R3 = 0
        elif dR12 and not dR23:
            # x = alfa
            # P4 = C1 + [R2,0], where C2 = C1 + (R1-R2)*[sin(x),cos(x)], where C1 = P1 + [0,-R1]
            # P4x = P1x+(R1-R2)sin(x)+R2
            # System of equations to solve:
            # 0=Asin(x)+B1 -> x = asin(-B1/A), where
            A = R1-R2
            B1 = P[1][0] + R2 - P[4][0]
            alfa = np.arctan(-B1/A)*180/np.pi
            beta = 90 - alfa
            gamma = 0

        elif dR12 and dR23:
            # x = alfa, y = alfa+beta
            # P4 = C3 + [R2,0], where C3 = C2 + (R2-R3)*[sin(y),cos(y)],  C2 = C1 + (R1-R2)*[sin(x),cos(x)], C1 = P1 + [0,-R1]
            # P4x = P1x +0+(R1-R2)*sin(x)+(R2-R3)*sin(y)+R3
            # P4y = P1y-R1+(R1-R2)*cos(x)+(R2-R3)*cos(y)+0
            # System of equations to solve:
            # 0=Asin(x)+Bsin(y)+C1
            # 0=Acos(x)+Bcos(y)+C2 where x=alfa, y=alfa+beta
            A = R1-R2
            B = R2-R3
            C1 = P[1][0]+R3-P[4][0]
            C2 = P[1][1]-R1-P[4][1]
            # Equations to express y:
            # sin(y)=(-Asin(x)-C1)/B
            # cos(y)=(-Acos(x)-C2)/B
            # From sin(y)**2+cos(y)**2 = 1 -> A**2+2AC1​sin(x)+C1**2​+2AC2​cos(x)+C2**2​=B2 ->
            # C1sin(x)+C2cos(x) = K where K = (B**2-A**2-C1**2-C2**2)/2A
            # From Rsin(x+α) = C1sin(x)+C2cos(x) where  R = (C1**2+C2**2)**.5 and α = atan(C2/C1)
            # x = asin(K/R)-α
            K = (B**2 - A**2 - C1**2 - C2**2)/(2*A)
            R = (C1**2 + C2**2)**.5
            a = np.arctan(C2/C1)
            x = np.arcsin(K/R) - a
            y = np.arcsin((-A*np.sin(x) - C1)/B)
            alfa = x*180/np.pi
            beta = (y - x)*180/np.pi
            gamma = 90 - alfa - beta
            print(f'alfa = {alfa}, beta = {beta}')

        C[1] = P[1] + R1*ang(180)
        C[2] = C[1] + (R1-R2)*ang(      alfa)
        C[3] = C[2] + (R2-R3)*ang(      alfa + beta)
        C[5] = C[3] + T*ang(180)
        C[6] = C[5] + (R3-R2)*ang(180 - alfa - beta)
        C[7] = C[6] + (R2-R1)*ang(180 - alfa)

        P[2] = C[1] + R1*ang(      alfa)
        P[3] = C[2] + R2*ang(      alfa + beta)
        P[4] = C[3] + R3*ang(90) 
        P[5] = P[4] + T*ang(180)
        P[6] = C[5] + R3*ang(180 - alfa - beta)
        P[7] = C[6] + R2*ang(180 - alfa)
        P[8] = C[7] + R1*ang(180)

        PC[1] = C[1] + R1*ang(      alfa/2)
        PC[2] = C[2] + R2*ang(      alfa + beta/2)
        PC[3] = C[3] + R3*ang(      alfa + beta + gamma/2) 
        PC[5] = C[5] + R3*ang(180 - alfa - beta - gamma/2)
        PC[6] = C[6] + R2*ang(180 - alfa - beta/2)
        PC[7] = C[7] + R1*ang(180 - alfa/2)

        contour.append(L(P[0], P[1]))
        contour.append(LARC(P[1], P[2], PC[1], R1, C[1]))
        if dR12:
            contour.append(LARC(P[2], P[3], PC[2], R2, C[2]))
        if dR23:
            contour.append(LARC(P[3], P[4], PC[3], R3, C[3]))
        if T:
            contour.append(L(P[4], P[5]))
        if dR23:
            contour.append(LARC(P[5], P[6], PC[5], R3, C[5]))
        if dR12:
            contour.append(LARC(P[6], P[7], PC[6], R2, C[6]))
        contour.append(LARC(P[7], P[8], PC[7], R1, C[7]))
        contour.append(L(P[8], P[9]))
    else:
        contour.append(L(P[1], P[9]))

    # Starting point of distance D from hole normal to scallop side
    if SANG==0:
        PC[4] = P[4] 
    if 0<SANG<=gamma:
        PC[4] = C[5] + R3*ang(90 + SANG)
    elif gamma<SANG<=gamma+beta:
        PC[4] = C[6] + R2*ang(90 + SANG)
    elif gamma+beta<SANG<=90:
        PC[4] = C[7] + R1*ang(90 + SANG)
    PC[10] = PC[4] + D*ang(90+SANG)
    print(f'PC10: {PC[10]}')

    # horizontal line P9-P10
    P[10][1] = P[9][1] 
    P[10][0] = PC[10][0] - (PC[10][1] - P[10][1])*np.tan(SANG*np.pi/180)
    contour.append(L(P[9], P[10]))

    # x = length of P10-PC11 vector, y = fi (angle <) P0,CO,C11)
    # P12 = P10 + x*[sin(SANG),cos(SANG)] + Rfil*[sin(90+SANG),cos(90+SANG)] + Rfil*[sin(y),cos(y)] = Ro*[sin(y),cos(y)]
    # System of equations to solve:
    # 0=A1x+Bsin(y)+C1
    # 0=A2x+Bcos(y)+C2, where 
    A1 = np.sin(SANG*np.pi/180)
    A2 = np.cos(SANG*np.pi/180)
    B  = Rfil-Ro
    C1 = P[10][0] + Rfil*np.sin((90+SANG)*np.pi/180)
    C2 = P[10][1] + Rfil*np.cos((90+SANG)*np.pi/180)
    # Isolate x: A1x = −Bsin(y)−C1, A2x = −Bcos(y)−C2 -> x=x -> A1cos(y)−A2sin(y) = (A1C2−A2C1)/B
    # Using identity A1cos(y)−A2sin(y) = Rcos(y+α), where R=(A1**2+A2**2)**.5=1
    a = np.arctan(A2/A1)
    y = np.arccos((A2*C1-A1*C2)/B) - a
    x = (-B*np.sin(y)-C1)/A1
    fi = y*180/np.pi
    print(f'x: {x}, fi: {fi}')

    P[11] = P[10] + x*ang(SANG)
    C[11] = P[11] + Rfil*ang(90+SANG)
    P[12] = C[11] + Rfil*ang(fi)
    P[13] = C[0]  + Ro*ang(teta)
    P[14] = C[0]  + Rext*ang(teta)
    P[15] = C[0]  + Rext*ang(0)

    PC[11] = C[11] + Rfil*ang((-90+SANG+fi)/2)
    PC[12] = C[0]  + Ro*ang((fi+teta)/2)   
    PC[14] = C[0]  + Rext*ang(teta/2)

    contour.append(L(P[10], P[11]))
    contour.append(LARC(P[11], P[12], PC[11], Rfil, C[11]))
    contour.append(LARC(P[12], P[13], PC[12], Ro,   C[0]))
    contour.append(L(P[13], P[14]))
    contour.append(LARC(P[14], P[15], PC[14], Rext, C[0]))

    plt.plot(P.T[0],  P.T[1],  'b-')
    plt.plot(C.T[0],  C.T[1],  'g+')
    plt.plot(PC.T[0], PC.T[1], 'r.')
    plt.gca().set_aspect('equal')
    plt.show()

contour = scallop()
