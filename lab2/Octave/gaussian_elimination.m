A = [1.12, 6.31, 3.52, 5.31, 3.23;
     2.43, 5.23, 7.43, 6.54, 2.22;
     0.76, 4.98, 7.86, 4.00, 7.43;
     9.99, 4.33, 5.44, 3.45, 2.45;
     6.43, 5.44, 3.23, 1.23, 2.67];

b = [1.23; 2.12; 7.54; 8.55; 3.45];

C = [A b]

n = size(C, 1);

for i = 1:n
    for j = i+1:n
        factor = C(j, i) / C(i, i);
        C(j, :) = C(j, :) - factor * C(i, :);
    end
end

disp("Upper Triangular Matrix:");
disp(C);