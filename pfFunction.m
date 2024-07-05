function [Pg] = pfFunction(Kij,delta1,delta2)
Pg = Kij*sin(delta1-delta2);
end