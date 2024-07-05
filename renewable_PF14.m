% 时间区间 [0, 1]
t = linspace(0, 6, 601);

%data = load('expanded_windPower.mat');

folderPath = './windData_300_mat';

files = dir(fullfile(folderPath, 'result_windpower_1.mat'));

for idx = 1:length(files)
    dataPath = fullfile(files(idx).folder, files(idx).name);
    data = load(dataPath);
    
    wind_power = data.newData;
    bus_1_power = 5/12*(315 - wind_power) ;
    bus_3_power = 7/5 * bus_1_power;
    %0.25
    P1 = 71;
    % 315*0.4 = 126 * 1.5 = 189
    P2 = 189;

    % 0.35
    P3 = 85;
    
    P4 = 0;
    P5 = -90;
    P6 = 0;
    P7 = -100;
    P8 = 0;
    P9 = -125;
    
    r = 0.032;
    I = 0.4;
    
    K14 = -1713.55;
    K28 = -1593.9;
    K36 = -1712.24;
    K45 = -1093.3;
    K56 = -574.147;
    K67 = -1059.23;
    K78 = -1370.89;
    K89 = -610.266;
    K94 = -1207.9;
    
    
    theta_1 = 0;
    theta_2 = 0.1687;
    theta_3 = 0.0833;
    theta_4 = -0.0420;
    theta_5 = -0.0701;
    theta_6 = 0.0336;
    theta_7 = 0.0108;
    theta_8 = 0.0663;
    theta_9 = -0.0759;
    bus_phases = [theta_1,theta_2,theta_3,theta_4,theta_5,theta_6,theta_7,theta_8,theta_9];
    omega = pi;
    
    y_one = [theta_1,omega];
    y_two = [theta_2,omega];
    y_three = [theta_3,omega];
    
    
    usol1 = zeros(1, length(t));
    usol2 = zeros(1, length(t));
    usol3 = zeros(1, length(t));
    
    PF14 = zeros(1, length(t));
    PF28 = zeros(1, length(t));
    PF36 = zeros(1, length(t));

    for i = 1:length(t)-1
        % 使用ode45求解摆动方程，每次只前进一小步
        [t_span, y1] = ode45(@(t, y) getOneDelta(t, y, bus_1_power(i), r, I, K14, bus_phases(end,4)), [t(i), t(i+1)], y_one);
        [t_span, y2] = ode45(@(t, y) getOneDelta(t, y, wind_power(i), r, I, K28, bus_phases(end,8)), [t(i), t(i+1)], y_two);    % 更新初始条件为最新的状态
        [t_span, y3] = ode45(@(t, y) getOneDelta(t, y, bus_3_power(i), r, I, K36, bus_phases(end,6)), [t(i), t(i+1)], y_three);
       
        y_one = y1(end, :).';
        y_two = y2(end, :).';
        y_three = y3(end, :).';
    
        usol1(i) = y1(end, 1);
        usol2(i) = y2(end, 1);
        usol3(i) = y3(end, 1);
    
        PG14_res = pfFunction(K14, y1(end, 1), bus_phases(end,4));
        PG28_res = pfFunction(K28, y2(end, 1), bus_phases(end,8));
        PG36_res = pfFunction(K36, y3(end, 1), bus_phases(end,6));
    
        PF14(i) = abs(PG14_res);
        PF28(i) = abs(PG28_res);
        PF36(i) = abs(PG36_res);
    
        % bus_phases = [y1(end, 1),y2(end, 1),y3(end, 1),y4(end, 1),y5(end, 1),y6(end, 1),y7(end, 1),y8(end, 1),y9(end, 1)];
        %bus_phases = [usol1(i),usol2(i),usol3(i),usol4(i),usol5(i),usol6(i),usol7(i),usol8(i),usol9(i)];
    end
    % PF14 = PF14';
    % PFfolderPath = './PF14_300_mat';
    % resultFilename = fullfile(PFfolderPath, sprintf('result_PF14_%d.mat', idx));
    % save(resultFilename, 'PF14');  % 假设PF14是您想保存的结果变量
end



%data = load('./winddata/wind_power_1.mat');
%wind_data = struct2table(data.wind_data_table);
%wind_power = wind_data.PowerOutput;
%bus_1_power = 5/12*(315 - wind_power) ;
%bus_3_power = 7/5 * bus_1_power;




figure; % 创建一个新的图形窗口
plot(t(1:end), PF14, 'b-', 'DisplayName', 'PF14'); hold on; % 蓝色线表示 PG14
%plot(t(1:end), PF28, 'r-', 'DisplayName', 'PF28'); % 红色线表示 PG28
plot(t(1:end), PF36, 'g-', 'DisplayName', 'PF36'); % 绿色线表示 PG36

xlabel('Time (t)'); % x轴标签
ylabel('PF Values Over Time'); % 图表标题
legend('show'); % 显示图例
hold off; % 结束绘图