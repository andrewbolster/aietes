%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Copyright (c) 2013 Plextek Limited, Great Chesterford, England.
%   All Rights Reserved
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   THIS FILE IS VERSION CONTROLLED USING SVN. DO NOT HAND EDIT THESE FIELDS
%   ----------------------------------------------------------------------------
%   $HeadURL$
%   $1$
%   $Aled Catherall$
%   $Date$
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   File Description
%   ================
% 
%   Script to predict the positional drift of a Unmanned underwater vehicle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function UUV_position_drift_model()
clear all

% Monte-Carlo Properties
seed =1; % seed for random number generator
s = RandStream('mt19937ar','Seed',seed); %set the seed parameter for random number generator
numloops = 100; % number of monte-carlo iterations

% Mission properties
Duration = 8*60*60;  % Mission length in seconds
dt = 1;       % time step (1 second for intertial nav)
x0 = 0;       % start coordinate
y0 = 0;       % start y coordinate
n = Duration/dt;  % number of time-steps in mission
mission_type = 2; % options: 0 = straight line 1 = big circle, 2 = lawn mower pattern



% monte-Carlo loop iteration
    error_sum2 = x0*ones(1,n);  % Initialise estimated x position

    for jj = 1:numloops    
                
      % initialise and reset these values each iteration of monte-carlo simulation     
      x_est = x0*ones(1,n);  % Initialise  x position array
      y_est = y0*ones(1,n);  % Initialise  y position array
      x_true = x0*ones(1,n);  % Initialise estimated x position array
      y_true = y0*ones(1,n);  % Initialise estimated y position array

      bearing_true = 0; %initialise initial bearing
      bearing_est = 0;  %initialise initialis eestimated bearing
      
  
      
      t = 0;            % initialise time
      
      % Bias terms for sensors   
      bias = 0;         % initialise sensor biases
      bias_along = 0;
      bias_across = 0;
      Gyro_scale_rand_term = randn(s,1,1);% Random Gaussian error onto bias - constant during a monte-carlo loop, but different between them
      DVL_scale_rand_term  = randn(s,1,1);  %Starting value of scale error  
      
      % update veocity and bearing of uuv
      [v_along v_across bearing_true] = update_uuv_position(t,mission_type);
      % Print to screen every 20th iteration of Monte-Carlo
      if mod(jj,20) == 0
        jj
      end


    %%%%   start the simulated run
       for ii = 2:n
 
            t = t + dt;
            
            [v_along_new v_across_new bearing_true_new] = update_uuv_position(t,mission_type);
            
            
       
            bearing_true_rate = (bearing_true_new - bearing_true)/dt;
            bearing_true = bearing_true_new;
            
            % obtain parameters from Doppler Velocity Log and Gyro
            [v_along_est , v_across_est, bias_along,bias_across] = DVL(v_along, v_across, s, dt, bias_along,bias_across,DVL_scale_rand_term,ii);
            [bearing_est_rate, bias] = GYRO(bearing_true_rate, s, dt,bias,Gyro_scale_rand_term,ii);
            
         
           
            % update estimated bearing
            bearing_est = bearing_est + bearing_est_rate*dt;
            
            % update estimated UUV position (first order Taylor finite difference)         
            x_est(ii) = x_est(ii-1) + v_along_est*dt*cos(bearing_est) + v_across_est*dt*sin(bearing_est); 
            y_est(ii) = y_est(ii-1) + v_along_est*dt*sin(bearing_est) + v_across_est*dt*cos(bearing_est);
            
            % update true position(first order Taylor finite difference) 
            x_true(ii) = x_true(ii-1) + v_along*dt*cos(bearing_true) + v_across*dt*sin(bearing_true);
            y_true(ii) = y_true(ii-1) + v_along*dt*sin(bearing_true) + v_across*dt*cos(bearing_true); 
            
            % calculate error
            error_sum2(ii) = error_sum2(ii) + (x_est(ii)-x_true(ii)).^2+(y_est(ii)-y_true(ii)).^2;
          
            
           
       end % end of ii loop
       % calculate rms error
       error_rms = sqrt(error_sum2/numloops);
     
     
   

    end % end of jj loop
    % Plot rms error
     figure(1)
     hold on
     plot(dt*(1:n),error_rms,'-*k')
     xlabel('time / s');
     ylabel(' error / m');
     figure(2)
     plot(x_true,y_true,'b')
     xlabel('x / m');
     ylabel('y / m');
     hold on
     plot(x_est,y_est,'r')
 
     
end   % END FUNCITON


function [v_along_est, v_across_est, bias_along, bias_across] = DVL(v_along_true, v_across_true, s, dt, bias_along,bias_across,DVL_scale_rand_term,count)
  
% Outputs estimated velocities and biases (along and across track)
% this function should only be called once every second. !!!!!!!!!!!!!
% the white noise is a per-measurment term, not a per root-hour
  % scale
  DVL_scale_accuracy = 1e-3;  
  DVL_scale_error = DVL_scale_accuracy * DVL_scale_rand_term; % add the random term which is constant each monte-carlo loop
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % white noise
  std_DVL_wn = 0.004;  % metres per second 
  ran = randn(s,1,1);
  white_noise_along =  std_DVL_wn*ran;  
  ran = randn(s,1,1);
  white_noise_across =  std_DVL_wn*ran; 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   % Coloured noise (as a Markov process)
  std_DVL_cn_al = 0.0041; % meters per second
  std_DVL_cn_ac = 0.001;  % meters per second
  tc_DVL_cn = 1800;     % time constant for markov process
 
    if count == 2  % on first iteration of simulation, set the bias
        ran = randn(s,1,1);  % generate random number (0,1)  = mean 0, std 1;
        bias_along = std_DVL_cn_al*ran;
        ran = randn(s,1,1);  % generate random number (0,1)  = mean 0, std 1;
        bias_across = std_DVL_cn_ac*ran;
    else
     bias_along = Markov_process(bias_along,tc_DVL_cn,std_DVL_cn_al,dt,s);
     bias_across = Markov_process(bias_across,tc_DVL_cn,std_DVL_cn_ac,dt,s);
    end

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % Add all errors to the true velocity to give estimated velocity
  v_along_est =  v_along_true*(1+DVL_scale_error)  +  bias_along  + white_noise_along;
  v_across_est = v_across_true*(1+DVL_scale_error) +  bias_across + white_noise_across;
  
 
end



function [bearing_est_rate, bias] = GYRO(bearing_true_rate,s,dt,bias,Gyro_scale_rand_term,count)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Outputs estimated bearing rate and gyro bias
% Gyroscope properties
% you can call this function as often as you want

% Scale error
Gyro_scale_accuracy = 1e-4; 
Gyro_scale_error = Gyro_scale_accuracy*Gyro_scale_rand_term;% add the random term which is constant each monte-carlo loop

% random white noise
ran = randn(s,1,1);  % generate random number (0,1)  = mean 0, std 1;
std_gyro_wn = 0.0025;  % degrees per root hour
std_gyro_wn = std_gyro_wn/3600;
std_gyro_wn = std_gyro_wn * pi/180;
q1 = std_gyro_wn^2;
white_noise_term = sqrt(q1*dt)*ran;

% Coloured noise (also known as bias)
std_gyro_cn = 0.0035; % degrees per hour
std_gyro_cn = std_gyro_cn/3600; % degrees per second
std_gyro_cn = std_gyro_cn*pi/180; % radians per second
tc_gyro_cn = 3600;    % time constant markov process
if count == 2
    ran = randn(s,1,1);  
    bias = std_gyro_cn*ran; % initial value at t = 0
else
    bias = Markov_process(bias,tc_gyro_cn,std_gyro_cn,dt,s); 
end

%calculate bearing rate
bearing_est_rate = bearing_true_rate*(1+Gyro_scale_error) + bias  + white_noise_term/dt;
end

function [bias] = Markov_process(bias,tc,std,dt,s)
% standard markov process
ran = randn(s,1,1);
q = 2*std*std/tc;
bias = bias * exp(-dt/tc) + sqrt(0.5*q*tc*(1 - exp(-2*dt/tc)))*ran;

end


function [v_along_new v_across_new bearing_true_new] = update_uuv_position(t,mission_type);

    switch mission_type
    
        case 0  % straight line
            v_along_new = 1.5;  % metres per second
            v_across_new = 0.0;
            bearing_true_new = 0.0;

        case 1  % big circle

            v_along_new = 1.5;
            v_across_new = 0.0;
            bearing_true_new = 0.002*t; %500 seconds per %pi radians  

        case 2   % lawn mower pattern
            t_leg = 1500;
            t_turn = 125;
            v_along_new = 1.5;  % metres per second     
            v_across_new = 0.0;
            if t < 750      % traverse to survey site      

                bearing_true_new = 0.0;
            elseif t<27600

                % start lawnmower pattern
                % 900s straight line, then 180 degree turn taking 100 s
                tt = t - 750;
                leg_no = floor(tt/t_leg);
                parity = mod(leg_no,2);
                leg_time = mod(tt,t_leg);

                if leg_time <t_leg - t_turn
                    bearing_true_new = 0 + pi*parity; 

                else
                        if parity == 0
                            bearing_true_new = (leg_time-t_leg+t_turn)*pi/t_turn;

                        else
                            bearing_true_new = pi-(leg_time-t_leg+t_turn)*pi/t_turn;
                        end
                end
                
            elseif t<27600+t_turn
                tt = t - 27600;
                 bearing_true_new = pi+0.4*tt*pi/t_turn;
                 
            else
                bearing_true_new = -pi+1.1;
            end
            
            
            
    end      
  
end