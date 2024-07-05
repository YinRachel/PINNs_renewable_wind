function dy = getOneDelta(t, y, P, r, I, K1,theta_1)
    dy = zeros(2, 1);    % 因为有两个一阶方程
    dy(1) = y(2);        % y(1) = δ, y(2) = ω
    dy(2) = P/I -(r / I) * y(2) + (K1 / I) * sin(y(1) - theta_1);  % ω的导数
end