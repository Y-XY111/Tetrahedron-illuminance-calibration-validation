function p=p_l_m(l,m,x)


if l==m 
    p=(-1).^(m).*fact2(2.*m-1).*(1-x.^2).^(m./2);
elseif l==m+1
    p=x.*(2.*m+1).*p_l_m(m,m,x);
else 
    p=(x.*(2.*l-1).*p_l_m (l-1,m,x)-(l+m-1).*p_l_m(l-2,m,x))./(l-m);
    
end


end

