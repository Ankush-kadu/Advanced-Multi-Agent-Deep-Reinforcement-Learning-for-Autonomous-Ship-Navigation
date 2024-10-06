import numpy as np
import module_kinematics as kin

def ode_options(scale=1, 
    wind_flag=False, wind_speed=0.0, wind_dir=180.0,
    wave_flag=False, wave_comp=None,
    current_flag=False, current_speed=0.0, current_dir=180.0,
    deltad_max=100, T_rud=1, T_prop=1, nd_max=0.1):

    options = {}

    options['scale'] = scale
    options['wind_flag'] = wind_flag
    options['wind_speed'] = wind_speed
    options['wind_dir'] = wind_dir

    options['wave_flag'] = wave_flag
    options['wave_comp'] = wave_comp

    options['current_flag'] = current_flag
    options['current_speed'] = current_speed
    options['current_dir'] = current_dir

    options['deltad_max'] = deltad_max
    options['T_rud'] = T_rud
    options['T_prop'] = T_prop
    options['nd_max'] = nd_max

    return options

def kcs_ode(t, v, delta_c, n_c, options):

    scale = options['scale']
    wind_flag = options['wind_flag']
    wind_speed = options['wind_speed']
    wind_dir = options['wind_dir']

    wave_flag = options['wave_flag']
    wave_comp = options['wave_comp']

    current_flag = options['current_flag']
    current_speed = options['current_speed']
    current_dir = options['current_dir']

    deltad_max = options['deltad_max']
    T_rud = options['T_rud']
    T_prop = options['T_prop']
    nd_max = options['nd_max']

    # Ship Geometry and Constants
    L = 230 / scale
    B = 32.2 / scale
    d_em = 10.8 / scale
    rho = 1025
    g = 9.80665

    Cb = 0.651
    Dsp = Cb * L * B * d_em
    Fn = 0.26
    U_des = Fn * np.sqrt(g * L)
    xG = -3.404 / scale
    kzzp = 0.25

    # Nondimensional State Space Variables

    up = v[0]
    vp = v[1]
    rp = v[5]
    xp = v[6]
    yp = v[7]
    q = v[9:13]
    eul = kin.quat_to_eul(q)
    psi = eul[2]
    delta = v[13]
    n_prop = v[14]

    # Derived kinematic variables
    b = np.arctan2(-vp, up)     # Drift angle


    # Surge Hydrodynamic Derivatives in non-dimensional form
    X0 = -0.0167
    Xbb = -0.0549
    Xbr_minus_my = -0.1084
    Xrr = -0.0120
    Xbbbb = -0.0417

    # Sway Hydrodynamic Derivatives in non-dimensional form
    Yb = 0.2252
    Yr_minus_mx = 0.0398
    Ybbb = 1.7179
    Ybbr = -0.4832
    Ybrr = 0.8341
    Yrrr = -0.0050

    # Yaw Hydrodynamic Derivatives in non-dimensional form
    Nb = 0.1111
    Nr = -0.0465
    Nbbb = 0.1752
    Nbbr = -0.6168
    Nbrr = 0.0512
    Nrrr = -0.0387

    # Non-dimensional Surge Hull Hydrodynamic Force
    Xp_H = X0 * (up ** 2) \
        + Xbb * (b ** 2) + Xbr_minus_my * b * rp \
        + Xrr * (rp ** 2) + Xbbbb * (b ** 4)

    # Non-dimensional Sway Hull Hydrodynamic Force
    Yp_H = Yb * b + Yr_minus_mx * rp + Ybbb * (b ** 3) \
        + Ybbr * (b ** 2) * rp + Ybrr * b * (rp ** 2) \
        + Yrrr * (rp ** 3)

    # Non-dimensional Yaw Hull Hydrodynamic Moment
    Np_H = Nb * b + Nr * rp + Nbbb * (b ** 3) \
        + Nbbr * (b ** 2) * rp + Nbrr * b * (rp ** 2) \
        + Nrrr * (rp ** 3)

    # Propulsion Force Calculation

    # The value self propulsion RPM is taken from Yoshimura's SIMMAN study
    # Analysis of steady hydrodynamic force components and prediction of
    # manoeuvering ship motion with KVLCC1, KVLCC2 and KCS
    # n_prop = 115.5 / 60
    Dp = 7.9 / scale
    wp = 1 - 0.645  # Effective Wake Fraction of the Propeller
    tp = 1 - 0.793  # Thrust Deduction Factor

    if n_prop > 1e-3:

        J = (up * U_des) * (1 - wp) / (n_prop * Dp)     # Advance Coefficient

        a0 = 0.5228
        a1 = -0.4390
        a2 = -0.0609
        Kt = a0 + a1 * J + a2 * (J ** 2)        # Thrust Coefficient

        # Dimensional Propulsion Force
        X_P = (1 - tp) * rho * Kt * (Dp ** 4) * (n_prop ** 2)
        
    else:
        J = 1.0
        Kt = 0.0
        X_P = 0.0

    # Non-dimensional Propulsion Force
    Xp_P = X_P / (0.5 * rho * L * d_em * (U_des ** 2))

    # Rudder Force Calculation
    A_R = L * d_em / 54.86
    Lamda = 2.164
    f_alp = 6.13 * Lamda / (2.25 + Lamda)

    eps = 0.956
    eta = 0.7979
    kappa = 0.633
    xp_P = -0.4565  # Assuming propeller location is 10 m ahead of AP (Rudder Location)
    xp_R = -0.5

    b_p = b - xp_P * rp

    if b_p > 0:
        gamma_R = 0.492
    else:
        gamma_R = 0.338

    lp_R = -0.755
    if J != 0.0:
        up_R = eps * (1 - wp) * up * np.sqrt(eta * (1 + kappa * \
            (np.sqrt(1 + 8 * Kt /(np.pi * (J ** 2)) ) - 1)) ** 2 + (1 - eta))

        vp_R = gamma_R * (vp + rp * lp_R)
    else:
        up_R = eps * (1 - wp) * up
        vp_R = gamma_R * (vp + rp * lp_R)

    Up_R = np.sqrt(up_R ** 2 + vp_R ** 2)
    alpha_R = delta  - np.arctan2(-vp_R, up_R)

    F_N = A_R / (L * d_em) * f_alp * (Up_R ** 2) * np.sin(alpha_R)

    tR = 1 - 0.742
    aH = 0.361
    xp_H = -0.436

    Xp_R = - (1 - tR) * F_N * np.sin(delta)
    Yp_R = - (1 + aH) * F_N * np.cos(delta)
    Np_R = - (xp_R + aH * xp_H) * F_N * np.cos(delta)

    # Coriolis terms

    mp = Dsp / (0.5 * (L ** 2) * d_em)
    xGp = xG / L

    Xp_C = mp * vp * rp + mp * xGp * (rp ** 2)
    Yp_C = -mp * up * rp
    Np_C = -mp * xGp * up * rp
    
    if wind_flag:
        Vw = wind_speed  # wind speed
        betaw = wind_dir * (np.pi/180)  # wind direction
        uw = Vw * np.cos(betaw - psi)
        vw = Vw * np.sin(betaw - psi)
        urw = up - uw
        vrw = vp - vw
        Uwr = (urw ** 2 + vrw ** 2) ** 0.5
        gammaw = np.arctan2(-vrw, -urw)

        rhow = 1025
        rhoa = 1.225
        Ax = 0.1064
        Ay = 0.7601
        
        Cwx = 1 * np.cos(gammaw)
        Cwy = 1 * np.sin(gammaw)
        Cwpsi = 0.5 * np.sin(gammaw)

        Xp_W = (Ax * Cwx * Uwr * abs(Uwr)) * rhoa / rhow
        Yp_W = (Ay * Cwy * Uwr * abs(Uwr)) * rhoa / rhow
        Np_W = (Ay * Cwpsi * Uwr * abs(Uwr)) * rhoa / rhow
    else:
        Xp_W = 0.0
        Yp_W = 0.0
        Np_W = 0.0
    
    # Net non-dimensional force and moment computation
    Xp = Xp_H + Xp_R + Xp_C + Xp_W + Xp_P
    Yp = Yp_H + Yp_R + Yp_C + Yp_W
    Np = Np_H + Np_R + Np_C + Np_W

    # Net force vector computation
    X = Xp
    Y = Yp
    N = Np

    # Added Mass and Mass Moment of Inertia (from MDLHydroD)
    mxp = 1790.85 / (0.5 * ((L * scale) ** 2) * (d_em * scale))
    myp = 44324.18 / (0.5 * ((L * scale) ** 2) * (d_em * scale))
    Jzzp = 140067300 / (0.5 * ((L * scale) ** 4) * (d_em * scale))
    Izzp = mp * (kzzp ** 2) + mp * (xGp ** 2)

    Mmat = np.zeros((3, 3))

    Mmat[0, 0] = mp + mxp
    Mmat[1, 1] = mp + myp
    Mmat[2, 2] = Izzp + Jzzp
    Mmat[1, 2] = mp * xGp
    Mmat[2, 1] = mp * xGp

    Mmatinv = np.linalg.inv(Mmat)

    tau = np.array([X, Y, N])

    vel_der = Mmatinv @ tau

    # Derivative of state vector
    vd = np.zeros(15)

    vd[0:2] = vel_der[0:2]
    vd[5] = vel_der[2]
    vd[6] = up * np.cos(psi) - vp * np.sin(psi)
    vd[7] = up * np.sin(psi) + vp * np.cos(psi)
    w = np.array([0, 0, rp])
    vd[9:13] = kin.quat_rate(q, w)

    # T_rud = 1     # Corresponds to a time constant of 1 * L / U_des = 20 seconds
    deltad = (delta_c - delta) / T_rud
    deltad_max = deltad_max * np.pi / 180 * (L / U_des)   # Maximum rudder rate of 5 degrees per second

    # Rudder rate saturation
    if np.abs(deltad) > deltad_max:
        deltad = np.sign(deltad) * deltad_max

    # T_prop = 1     # Corresponds to a time constant of 1 * L / U_des = 20 seconds
    nd_prop = (n_c - n_prop) / T_prop
    # print(nd_prop)
    # nd_max = 0.1

    # Rudder rate saturation
    if np.abs(nd_prop) > nd_max:
        nd_prop = np.sign(nd_prop) * nd_max

    vd[13] = deltad
    vd[14] = nd_prop

    # print(vd)

    return vd

def onrt_ode(t, ss, delta_c, n_c, options, euler_angle_flag=False):

    scale = options['scale']
    wind_flag = options['wind_flag']
    wind_speed = options['wind_speed']
    wind_dir = options['wind_dir']

    wave_flag = options['wave_flag']
    wave_comp = options['wave_comp']

    current_flag = options['current_flag']
    current_speed = options['current_speed']
    current_dir = options['current_dir']

    deltad_max = options['deltad_max']
    T_rud = options['T_rud']
    T_prop = options['T_prop']
    nd_max = options['nd_max']

    # All coefficients imported below are non-dimensional

    M_RB = options['M_RB']
    M_A = options['M_A']
    Dl = options['Dl']
    K = options['K']
    M_inv = options['M_inv']

    Xuu = options['X_u_au']
    Yvv = options['Y_v_av']
    Yvr = options['Y_v_ar']
    Yrv = options['Y_r_av']
    Yrr = options['Y_r_ar']
    Nvv = options['N_v_av']
    Nvr = options['N_v_ar']
    Nrv = options['N_r_av']
    Nrr = options['N_r_ar']

    D_prop = options['D_prop']
    pow_coeff_port = options['pow_coeff_port']
    pow_coeff_stbd = options['pow_coeff_stbd']

    A_R = options['rudder_area']
    Lamda = options['rudder_aspect_ratio']

    wp = options['wp']
    tp = options['tp']
    xp_R = options['xp_R']
    tR = options['tR']
    aH = options['aH']
    xp_H = options['xp_H']
    eta = options['eta']
    eps = options['eps']
    X_by_Dp = options['X_by_Dp']
    lp_R = options['lp_R']
    gamma_R1 = options['gamma_R1']
    gamma_R2 = options['gamma_R2']

    L = options['L']
    U_des = options['U_des']

    # Nondimensional State Space Variables

    up = ss[0]; vp = ss[1]; wp = ss[2]
    pp = ss[3]; qp = ss[4]; rp = ss[5]
    xp = ss[6]; yp = ss[7]; zp = ss[8]
    
    if euler_angle_flag:
        eul = ss[9:12]
    else:
        quat = ss[9:13]
        eul = kin.quat_to_eul(quat)
    
    phi = eul[0]; theta = eul[1]; psi = eul[2]

    if euler_angle_flag:
        rud_indx = 12
        prop_indx = 13
    else:
        rud_indx = 13
        prop_indx = 14
    
    delta = ss[rud_indx]; n_prop = ss[prop_indx]

    v1 = ss[0:3]; v2 = ss[3:6]

    # Coriolis force calculation

    M11 = M_RB[0:3][:, 0:3]
    M12 = M_RB[0:3][:, 3:6]
    M21 = M_RB[3:6][:, 0:3]
    M22 = M_RB[3:6][:, 3:6]

    C_RB = np.zeros((6,6))
    C_RB[0:3][:, 3:6] = -kin.Smat(M11 @ v1 + M12 @ v2)
    C_RB[3:6][:, 0:3] = -kin.Smat(M11 @ v1 + M12 @ v2)
    C_RB[3:6][:, 3:6] = -kin.Smat(M21 @ v1 + M22 @ v2)

    A11 = M_A[0:3][:, 0:3]
    A12 = M_A[0:3][:, 3:6]
    A21 = M_A[3:6][:, 0:3]
    A22 = M_A[3:6][:, 3:6]

    C_A = np.zeros((6,6))
    C_A[0:3][:, 3:6] = -kin.Smat(A11 @ v1 + A12 @ v2)
    C_A[3:6][:, 0:3] = -kin.Smat(A11 @ v1 + A12 @ v2)
    C_A[3:6][:, 3:6] = -kin.Smat(A21 @ v1 + A22 @ v2)    

    tau_C = -(C_RB @ ss[0:6] + C_A @ ss[0:6])

    # Damping force calculation

    Dnl = np.zeros(6)
    Dnl[0] = -Xuu * up * np.abs(up)
    Dnl[1] = -(Yvv * vp * np.abs(vp) + Yrv * rp * np.abs(vp) + Yvr * vp * np.abs(rp) + Yrr * rp * np.abs(rp))
    Dnl[5] = -(Nvv * vp * np.abs(vp) + Nrv * rp * np.abs(vp) + Nvr * vp * np.abs(rp) + Nrr * rp * np.abs(rp))

    tau_D = -(Dl @ ss[0:6]) - Dnl

    # Stiffness force calculation

    xi = np.zeros(6)
    xi[0:3] = ss[6:9]    
    xi[3:6] = eul
    tau_K = -(K @ xi)

    # Propeller force calculation

    X_port = pow_coeff_port[0] * ((up * (1 - wp)) **2) * (D_prop **2) \
                + pow_coeff_port[1] * (up * (1 - wp)) * (n_prop * (D_prop ** 3)) \
                + pow_coeff_port[2] * ((n_prop ** 2) * (D_prop ** 4))
    
    X_stbd = pow_coeff_stbd[0] * ((up * (1 - wp)) **2) * (D_prop **2) \
                + pow_coeff_stbd[1] * (up * (1 - wp)) * (n_prop * (D_prop ** 3)) \
                + pow_coeff_stbd[2] * ((n_prop ** 2) * (D_prop ** 4))

    tau_P = np.zeros(6)
    tau_P[0] = 2 * (1 - tp) * (X_port + X_stbd)

    # Rudder force calculation
    
    kappa_R = 0.5 + 0.5 * X_by_Dp / (X_by_Dp + 0.15)

    pow_coeff = 0.5 * (pow_coeff_port + pow_coeff_stbd)    

    up_prop = up * (1 - wp)
    Kt_up2_by_J2 = pow_coeff[0] * (up_prop ** 2) + pow_coeff[1] * n_prop * D_prop * up_prop + pow_coeff[2] * (n_prop * D_prop) ** 2
    up_R = eps * np.sqrt( eta * (up_prop + kappa_R * (np.sqrt(up_prop ** 2 + 8/np.pi * Kt_up2_by_J2) - up_prop)) ** 2 + (1 - eta) * up_prop ** 2 )

    b = np.arctan2(-vp, up)     # Drift angle
    b_p = b - (-X_by_Dp * D_prop) * rp
    
    # gamma_R (formula from from KCS structure - values modified)
    if b_p > 0:
        gamma_R = gamma_R1 # 0.6
    else:
        gamma_R = gamma_R2 # 0.3
    
    vp_R = gamma_R * (vp + rp * lp_R)

    # Resultant fluid speed at rudder
    Up_R = np.sqrt(up_R ** 2 + vp_R ** 2)

    # Effective inflow angle to rudder
    alpha_R = delta  - np.arctan2(-vp_R, up_R)

    # Rudder lift gradient coefficient
    f_alp = 6.13 * Lamda / (2.25 + Lamda)

    # Rudder normal force
    F_N = A_R * f_alp * (Up_R ** 2) * np.sin(alpha_R)

    Xp_R = - (1 - tR) * F_N * np.sin(delta)
    Yp_R = - (1 + aH) * F_N * np.cos(delta)
    Np_R = - (xp_R + aH * xp_H) * F_N * np.cos(delta)

    tau_R = np.zeros(6)
    tau_R[0] = Xp_R
    tau_R[1] = Yp_R
    tau_R[5] = Np_R

    # Total force calculation

    tau = tau_C + tau_D + tau_K + tau_P + tau_R

    # print(tau_P[0], tau_D[0], Xuu, up)

    # Derivative of velocity vector

    veld = M_inv @ tau

    # Derivative of state vector
    if euler_angle_flag:
        n = 14
    else:
        n = 15

    ssd = np.zeros(n)
    ssd[0:6] = veld

    if euler_angle_flag:
        ssd[6:9] = kin.eul_to_rotm(eul) @ v1
        ssd[9:12] = kin.eul_rate(eul, v2)
    else:
        ssd[6:9] = kin.quat_to_rotm(quat) @ v1
        ssd[9:13] = kin.quat_rate(quat, v2)
    
    # Rudder dynamics

    deltad = (delta_c - delta) / T_rud
    deltad_max = deltad_max
    # Rudder rate saturation
    if np.abs(deltad) > deltad_max:
        deltad = np.sign(deltad) * deltad_max

    # Propeller dynamics

    nd_prop = (n_c - n_prop) / T_prop
    # Propeller speed rate saturation
    if np.abs(nd_prop) > nd_max:
        nd_prop = np.sign(nd_prop) * nd_max

    ssd[rud_indx] = deltad
    ssd[prop_indx] = nd_prop

    return ssd

