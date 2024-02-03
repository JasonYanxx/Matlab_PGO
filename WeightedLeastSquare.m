function [eWLSSolution,prn_res,ErrorECEF,G,eDeltaPos,eDeltaPr] = WeightedLeastSquare(GT_ecef, init_state, meas, meas_std, sv_pos)
% init_state: initial state x0 [P_ecef;C_b] without switch or [P_ecef; C_b;
% S_1;...S_M] with switch
% meas: pseudorange measurements in single epoch
% meas_std: std of pseudorange measurements
% sv_pos: corresponding satellites' positions in ECEF;
% sysidx: corresponding system index (3: GPS, 4: BDS)
% useSW: ture for using switch variables, false for not using switch
% variables
% output: 
% Sol: estimated state
% Hessian: Hessian matrix
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    % init to_be_estimated state
    eWLSSolution = init_state;
    bWLSConverge = false;
    count = 0;
    
    iNumSV = length(meas);
    
    OMGE_ = 7.2921151467E-5;
    CLIGHT_ = 299792458.0;
    eH_Matrix_ = zeros(iNumSV,4);
    Hessian_ = zeros(4,4);
    while ~bWLSConverge
        eH_Matrix = zeros(iNumSV,4);
        Hessian = zeros(4,4);
        eDeltaPr =  zeros(iNumSV,1);
        eDeltaPos = zeros(4,1);
    
        for idx = 1:length(sv_pos)
            % satellite position
            rs_x = sv_pos(idx,1);
            rs_y = sv_pos(idx,2);
            rs_z = sv_pos(idx,3);
            % user position
            rr_x = eWLSSolution(1,1);
            rr_y = eWLSSolution(2,1);
            rr_z = eWLSSolution(3,1);
            rcv_clb_bias = eWLSSolution(4,1);
            
            dGeoDistance = sqrt((rs_x - rr_x)^2 + (rs_y - rr_y)^2 +(rs_z - rr_z)^2);
             % if we used dgps corrected pseudorange meas, then the
            % following line is not needed.
%             dGeoDistance = dGeoDistance + OMGE_ * (rs_x*rr_y-rs_y*rr_x)/CLIGHT_;

            % construct Jacobian Matrix
            eH_Matrix(idx, 1) = -(rs_x - rr_x) / dGeoDistance;
            eH_Matrix(idx, 2) = -(rs_y - rr_y) / dGeoDistance;
            eH_Matrix(idx, 3) = -(rs_z - rr_z) / dGeoDistance;
            eH_Matrix(idx, 4) = 1.0;
            
            eDeltaPr(idx,1) = meas(idx) - dGeoDistance - rcv_clb_bias;
        end

        weight_matrix = inv(diag(meas_std.*meas_std));
%         weight_matrix = eye(iNumSV);
        eDeltaPos = inv(transpose(eH_Matrix) * weight_matrix * eH_Matrix) * ...
            transpose(eH_Matrix) * weight_matrix * eDeltaPr;
        eWLSSolution = eWLSSolution + eDeltaPos;
    
        for i = 1:3
            if (abs(eDeltaPos(i))>1e-4)
                bWLSConverge = false;
            else
                bWLSConverge = true;
            end
        end
        count = count+1;
        if count>25
            bWLSConverge = true;
        end
    
    end

    prn_res= eDeltaPr;
    G = eH_Matrix;
    ErrorECEF=sqrt((GT_ecef(1)-eWLSSolution(1))^2 + (GT_ecef(2)-eWLSSolution(2))^2 + (GT_ecef(3)-eWLSSolution(3))^2);
%     [lat, lon, height] = ecef2llh(eWLSSolution(1), eWLSSolution(2), eWLSSolution(3))
end


function dis = Euc_dis(A,B)
    % A, B is row vector
    dis = sqrt((A-B)*(A-B)');
end

function [lat, lon, height] = ecef2llh(x, y, z)
    % Define WGS84 ellipsoid parameters
    a = 6378137; % Semi-major axis (equatorial radius) in meters
    f = 1/298.257223563; % Flattening

    % Calculate eccentricity squared
    e2 = 2*f - f^2;

    % Calculate longitude
    lon = atan2(y, x);

    % Calculate distance from the Z-axis
    r = sqrt(x^2 + y^2);

    % Initial estimate of latitude
    lat = atan2(z, r);

    % Iteratively refine the estimate of latitude
    delta = 1;
    while delta > 1e-8
        N = a / sqrt(1 - e2 * sin(lat)^2);
        prevLat = lat;
        lat = atan2(z + N * e2 * sin(lat), r);
        delta = abs(lat - prevLat);
    end

    % Calculate height above the ellipsoid
    height = r / cos(lat) - N;

    % Convert latitude and longitude to degrees
    lat = rad2deg(lat);
    lon = rad2deg(lon);
end