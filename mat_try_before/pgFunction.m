function [Pg] = pgFunction(B12,V1,V2,delta1,delta2)
Pg = B12*V1*V2*sin(delta1-delta2);
end