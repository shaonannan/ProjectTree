h = fspecial('gaussian',4,3);
for i = 1:5:170
imgshow = reshape(VIavgbin_ra(i,:),3*4,5*4);
imgshow = conv2(imgshow,h,'same');
figure(1);
imagesc(imgshow');
axis equal;axis off;caxis([0,0.48]);
pause;
end