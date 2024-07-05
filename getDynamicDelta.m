function dy = getDynamicDelta(~, y, d, B12, V1, V2, P1, m, bus_phases)
    dy = zeros(2, 1);    % 因为有两个一阶方程
    dy(1) = y(2);        % y(1) = δ, y(2) = ω
    dy(2) = -(d / m) * y(2) - (B12 * V1 * V2 / m) * sin(y(1)-bus_phases) + P1 / m;  % ω的导数
    
end