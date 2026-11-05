function y_l_m=y_l_m(l, m, theta, phi)

if m>0
    y_l_m=sqrt(2).*k(l,m).*cos(m.*phi).*p_l_m(l,m,cos(theta));
elseif m<0
    y_l_m=sqrt(2).*k(l,-m).*sin(-m.*phi).*p_l_m(l,-m,cos(theta));
else
    y_l_m=k(l,0).*p_l_m(l,0,cos(theta));
end

end
