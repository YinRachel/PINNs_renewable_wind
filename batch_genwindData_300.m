% 时间范围和时间步长
t = 0:1:60;  % 从0到60秒，每秒生成一个数据点

for iteration = 1:300

    % 模拟风速数据，假设风速在5到20米/秒之间波动
    mean_wind_speed = 12;  % 平均风速
    std_dev_wind_speed = 4;  % 风速标准差
    wind_speed = mean_wind_speed + std_dev_wind_speed * randn(1, length(t));
    wind_speed = max(wind_speed, 3);  % 设置一个合理的最小值
    
    % 引入模拟的阵风效应
    gust_factor = 1.5;  % 阵风放大因子
    gust_timing = 20:30;  % 指定发生阵风的时间段
    wind_speed(gust_timing) = wind_speed(gust_timing) * gust_factor;
    wind_speed = movmean(wind_speed, 4);  % 使用移动平均平滑风速
    
    % 定义风力发电机的参数
    cut_in_speed = 4;  % 切入风速
    cut_out_speed = 25;  % 切出风速
    rated_speed = 12;  % 额定风速
    rated_power = 1890;  % 调整额定功率为1890 kW，提高输出匹配新的整体功率目标
    
    % 使用平滑后的风速重新计算发电功率
    power_output = zeros(1, length(t));
    for i = 1:length(t)
        ws = wind_speed(i);  % 使用平滑后的风速进行计算
        if ws < cut_in_speed
            power_output(i) = 0;  % 当风速低于切入风速时，输出功率为0
        elseif ws <= rated_speed
            power_output(i) = rated_power * (ws - cut_in_speed) / (rated_speed - cut_in_speed);
        else
            power_output(i) = rated_power;
        end
    end

    power_output = power_output/10;

    origData = power_output;
    origTime = t;
    newTime = linspace(0,60,601);
    newData = interp1(origTime,origData, newTime,'nearest');
    newWind = interp1(origTime,wind_speed,newTime,'nearest');
    newTime = newTime';
    newData = newData';
    newWind = newWind';
    windDataPath = './windData_300_mat';
    resultFileName = fullfile(windDataPath, sprintf('result_windpower_%d.mat',iteration));
    save(resultFileName, 'newTime','newWind',"newData")
end

% 绘图风速和功率输出
figure;
yyaxis left;
plot(t, wind_speed, 'b-', 'LineWidth', 2);
ylabel('Wind Speed (m/s)');
yyaxis right;
plot(t, power_output, 'r-', 'LineWidth', 2);
ylabel('Power Output (MW)');
xlabel('Time (s)');
title('Wind Power System Output Over 60 Seconds');
grid on;



%wind_data_table = table(t', power_output', 'VariableNames', {'Time',  'PowerOutput'});
%filename = 'wind_power_data.mat';
%save(filename, 'wind_data_table');