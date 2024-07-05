define_constants;  % 定义常数
mpc = loadcase('case9');  % 加载案例

% 时间区间 [0, 20]
t = linspace(0, 60, 601);
folderPath = './windData_300_mat';
time = t';
files = dir(fullfile(folderPath, '*.mat'));
tic;
% for idx = 1:length(files)
for idx = 1:50
    dataPath = fullfile(files(idx).folder, files(idx).name);
    data = load(dataPath);

    % 参数 d, B12, V1, V2, P1, m 需要给出它们的值
    d = 1;
    B12 = 1.63;
    V1 = 1;
    V2 = 1;
        
    wind_power = data.newData;
    wind_speed = data.newWind;
    bus_1_power = 5/12*(315 - wind_power) ;
    bus_3_power = 7/5 * bus_1_power;
    
    wind_power = wind_power/126;
    bus_1_power = bus_1_power/78.75;
    bus_3_power = bus_3_power/110.25;
    
    figure;
    plot(t,wind_power);
    
    
    P1m = 0.18;  % input 
    P2m = 0.18;
    P3m = 0.18;
    m = 0.4;
    
    % results = struct();
    % y_one = [0; pi]; % 初始条件
    % y_two = [0.1687; pi]; % 初始条件
    % y_three = [0.0833; pi]; % 初始条件
    % bus_phases = [-0.042, 0.0336, 0.0663];

    y_one = [2.8; 0]; % 初始条件
    y_two = [2.628; 0]; % 初始条件
    y_three = [2.6496; 0]; % 初始条件
    bus_phases = [-10.16, -23.8, -23.6];

    usol1 = zeros(1, length(t));
    usol2 = zeros(1, length(t));
    usol3 = zeros(1, length(t));
    usol4 = zeros(1, length(t));
    usol6 = zeros(1, length(t));
    usol8 = zeros(1, length(t));

    all_branch_pf_qf = zeros(9,length(t));
    all_bus_phases = zeros(9,length(t));
    
    PF14 = zeros(1, length(t));
    PF28 = zeros(1, length(t));
    PF36 = zeros(1, length(t));
    
    


    for i = 1:length(t)-1
        % 使用ode45求解摆动方程，每次只前进一小步
        [t_span, y1] = ode45(@(t, y) getDynamicDelta(t, y, d, B12, V1, V2, bus_1_power(i), m, bus_phases(end,1)), [t(i), t(i+1)], y_one);
        [t_span, y2] = ode45(@(t, y) getDynamicDelta(t, y, d, B12, V1, V2, wind_power(i), m, bus_phases(end,3)), [t(i), t(i+1)], y_two);
        [t_span, y3] = ode45(@(t, y) getDynamicDelta(t, y, d, B12, V1, V2, bus_3_power(i), m, bus_phases(end,2)), [t(i), t(i+1)], y_three);
    
        % [t_span, y1] = ode45(@(t, y) getDynamicDelta(t, y, d, B12, V1, V2, P1m, m, bus_phases(end,1)), [t(i), t(i+1)], y_one);
        % [t_span, y2] = ode45(@(t, y) getDynamicDelta(t, y, d, B12, V1, V2, P2m, m, bus_phases(end,3)), [t(i), t(i+1)], y_two);
        % [t_span, y3] = ode45(@(t, y) getDynamicDelta(t, y, d, B12, V1, V2, P3m, m, bus_phases(end,2)), [t(i), t(i+1)], y_three);
    
        % 更新初始条件为最新的状态
        y_one = y1(end, :).';
        y_two = y2(end, :).';
        y_three = y3(end, :).';
    
        % 存储每步结果
        usol1(i) = y1(end, 1);
        usol2(i) = y2(end, 1);
        usol3(i) = y3(end, 1);
        
        usol4(i) = bus_phases(end, 1);
        usol6(i) = bus_phases(end, 2);
        usol8(i) = bus_phases(end, 3);
        % 计算当前电网相位
        PG1 = pgFunction(B12, V1, V2, y1(end, 1), bus_phases(end,1));
        PG2 = pgFunction(B12, V1, V2, y2(end, 1), bus_phases(end,3));
        PG3 = pgFunction(B12, V1, V2, y3(end, 1), bus_phases(end,2));
        PF14(i) = abs(PG1);
        PF28(i) = abs(PG2);
        PF36(i) = abs(PG3);
        % 更新 bus_phases
        %bus_phases = get_bus_phases(PG1, PG2, PG3);
        bus_phases(i+1, :) = get_bus_phases(PG1, PG2, PG3);
        all_branch_pf_qf(:,i) = allBranchPFFunc(PG1, PG2, PG3);
        all_bus_phases(:,i) = get_all_bus_phases(PG1, PG2, PG3);
    end
    windSpeed = wind_speed;
    windPower = wind_power;
    PF14 = PF14';
    PF28 = PF28';
    PF36 = PF36';
    
    usol1 = usol1';
    usol2 = usol2';
    usol3 = usol3';
    usol4 = usol4';
    usol6 = usol6';
    usol8 = usol8';

    resultDataPath = './1_final_data';
    resultFileName = fullfile(resultDataPath, sprintf('all_result_%d.mat',idx));
    save(resultFileName,'time','windSpeed','windPower','PF14',"PF28",'PF36',"usol1",'usol2','usol3',"usol4",'usol6','usol8')

    figure;
    

    % 绘制第一个数据线 usol1
    subplot(3,1,1); % 创建一个包含三个子图的 3x1 网格，并在第一个位置绘制
    plot(t,PF14);
    title('PF14');
    % 
    % % 绘制第二个数据线 usol2
    subplot(3,1,2); % 在第二个位置绘制
    plot(t,PF28);
    title('PF28');
    % 
    % % 绘制第三个数据线 usol3
    subplot(3,1,3); % 在第三个位置绘制
    plot(t,PF36);
    title('PF36');
end
elapsed_time = toc;
disp(['代码的运行时间为：', num2str(elapsed_time), ' 秒']);

%plot(t, all_branch_pf_qf);
%plot(t, usol1);
%plot(t,all_bus_phases)
% 
% load_loss = sum(all_branch_pf_qf(:,100)-all_branch_pf_qf(:,101));
% disp(load_loss)

% usol1 = usol1';
% usol2 = usol2';
% usol3 = usol3';
% 
% figure;
%plot(t,PF14);

% 
% % 绘制第一个数据线 usol1
% subplot(3,1,1); % 创建一个包含三个子图的 3x1 网格，并在第一个位置绘制
% plot(t,PF14);
% title('PF14');
% % 
% % % 绘制第二个数据线 usol2
% subplot(3,1,2); % 在第二个位置绘制
% plot(t,PF28);
% title('PF28');
% % 
% % % 绘制第三个数据线 usol3
% subplot(3,1,3); % 在第三个位置绘制
% plot(t,PF36);
% title('PF36');
% 
% t=t';
% x1 = bus_phases(:,1);
% %usol1 = usol1';
% 
% % 移除前两行和最后一行
% t = t(3:end-1);
% x1 = x1(3:end-1);
% usol1 = usol1(3:end-1);
% save('gen_1_data.mat', 't', 'usol1','x1');
% 
% x2 = bus_phases(:,2);
% x2 = x2(3:end-1);
% usol2 = usol2(3:end-1);
% save('gen_2_data.mat', 't', 'usol2','x2');
% 
% x3 = bus_phases(:,2);
% x3 = x3(3:end-1);
% usol3 = usol3(3:end-1);
% save('gen_3_data.mat', 't', 'usol3','x3');

%title('Generator 1 Angle over Time');
%xlabel('Time (s)');
%ylabel('Angle (radians)');
