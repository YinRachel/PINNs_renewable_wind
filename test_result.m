data = load('./2_final_50/all_result_10.mat');
figure;
plot(time,windPower);
ylabel('Power(p.u.)');
xlabel('Time (s)');

% subplot(2,4,1);
% plot(time,PF14);
% title('PF14');
% 
% subplot(2,4,2);
% plot(time,usol1);
% title('usol1');
% 
% subplot(2,4,3);
% plot(time,PF28);
% title('PF28');
% 
% subplot(2,4,4);
% plot(time,usol2);
% title('usol2');
% 
% subplot(2,4,5);
% plot(time,PF36);
% title('PF36');
% 
% subplot(2,4,6);
% plot(time,usol3);
% title('usol3');
% 
% subplot(2,4,7);
% plot(time,windPower);
% title('windPower');

% 
% % 绘制第一个数据线 usol1
% subplot(2,2,1); % 创建一个包含三个子图的 3x1 网格，并在第一个位置绘制
% plot(time,usol1);
% title('Bus 1 voltage angles');
% % 
% % % 绘制第二个数据线 usol2
% subplot(2,2,2); % 在第二个位置绘制
% plot(time,usol2);
% title('Bus 2 voltage angles');
% % 
% % % 绘制第三个数据线 usol3
% subplot(2,2,3); % 在第三个位置绘制
% plot(time,usol3);
% title('Bus 3 voltage angles');
% 
% subplot(2,2,4); % 在第三个位置绘制
% plot(time,windPower);
% title('wind power');



% subplot(2,1,1); % 创建一个包含三个子图的 3x1 网格，并在第一个位置绘制
% plot(time,usol1);
% % title('Bus 1 voltage angles');
% ylabel('Voltage angles(rad)');
% xlabel('Time (s)');
% subplot(2,1,2); % 在第三个位置绘制
% plot(time,windPower);
% % title('Wind power generation');
% ylabel('Power(p.u.)');
% xlabel('Time (s)');

% % 时间范围和时间步长
% t = 0:1:60;  % 从0到60秒，每秒生成一个数据点
% 
% % 模拟风速数据，假设风速在5到20米/秒之间波动
% mean_wind_speed = 12;  % 平均风速
% std_dev_wind_speed = 4;  % 风速标准差
% wind_speed = mean_wind_speed + std_dev_wind_speed * randn(1, length(t));
% wind_speed = max(wind_speed, 3);  % 设置一个合理的最小值
% 
% % 引入模拟的阵风效应
% gust_factor = 1.5;  % 阵风放大因子
% gust_timing = 20:30;  % 指定发生阵风的时间段
% wind_speed(gust_timing) = wind_speed(gust_timing) * gust_factor;
% wind_speed = movmean(wind_speed, 4);  % 使用移动平均平滑风速
% 
% % 定义风力发电机的参数
% cut_in_speed = 4;  % 切入风速
% cut_out_speed = 25;  % 切出风速
% rated_speed = 12;  % 额定风速
% rated_power = 1890;  % 调整额定功率为1890 kW，提高输出匹配新的整体功率目标
% 
% % 使用平滑后的风速重新计算发电功率
% power_output = zeros(1, length(t));
% for i = 1:length(t)
%     ws = wind_speed(i);  % 使用平滑后的风速进行计算
%     if ws < cut_in_speed
%         power_output(i) = 0;  % 当风速低于切入风速时，输出功率为0
%     elseif ws <= rated_speed
%         power_output(i) = rated_power * (ws - cut_in_speed) / (rated_speed - cut_in_speed);
%     else
%         power_output(i) = rated_power;
%     end
% end
% 
% power_output = power_output/10;
% 
% 
% 
% figure;
% yyaxis left;
% plot(t, wind_speed, 'b-', 'LineWidth', 2);
% ylabel('Wind Speed (m/s)');
% yyaxis right;
% plot(t, power_output, 'r-', 'LineWidth', 2);
% ylabel('Power Output (MW)');
% xlabel('Time (s)');
% title('Wind Power System Output Over 60 Seconds');
% grid on;