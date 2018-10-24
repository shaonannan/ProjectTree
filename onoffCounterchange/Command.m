%%% >>>->>>->>> Traditional On-/Off-counterchange >>>->>>->>>
h = fspecial('gaussian',4,3);
for i = 1:10:120
imgshow = reshape(VEavgbin_ra(i,:),3*4,5*4);
imgshow = conv2(imgshow,h,'same');
figure(1);
imagesc(imgshow');
axis equal;axis off;colormap('jet');caxis([0,0.62]); % 0,0.72<--50-->0.62
i
pause;
end

%%% >>>->>>->>> Off-only >>>->>>->>>
h = fspecial('gaussian',4,3);
for i = 1:10:61
imgshow = reshape(VEavgbin_ra(i,:),3*4,5*4);
imgshow = conv2(imgshow,h,'same');
figure(1);
imagesc(imgshow');
axis equal;axis off;colormap('jet');caxis([0,0.70]); % 0,0.68<--60-->0.60
i
pause;
end


%%% >>>->>>->>> On-only >>>->>>->>>
h = fspecial('gaussian',4,3);
for i = 1:8:61
imgshow = reshape(VEavgbin_ra(i,:),3*4,5*4);
imgshow = conv2(imgshow,h,'same');
figure(1);
imagesc(imgshow');
axis equal;axis off;colormap('jet');caxis([0,0.70]); % 0,0.68<--60-->0.60
i
pause;
end
h = fspecial('gaussian',4,3);
for i = 66:8:120
imgshow = reshape(VEavgbin_ra(i,:),3*4,5*4);
imgshow = conv2(imgshow,h,'same');
figure(1);
imagesc(imgshow');
axis equal;axis off;colormap('jet');caxis([0,0.64]); % 0,0.68<--60-->0.60
i
pause;
end

%%% >>>->>>->>> moving >>>->>>->>>
h = fspecial('gaussian',4,3);
for i = 1:5:61
imgshow = reshape(VEavgbin_ra(i,:),3*4,5*4);
imgshow = conv2(imgshow,h,'same');
figure(1);
imagesc(imgshow');
axis equal;axis off;colormap('jet');caxis([0,0.76]); % 0,0.68<--60-->0.60
i
pause;
end
h = fspecial('gaussian',4,3);
for i = 60:5:120
imgshow = reshape(VEavgbin_ra(i,:),3*4,5*4);
imgshow = conv2(imgshow,h,'same');
figure(1);
imagesc(imgshow');
axis equal;axis off;colormap('jet');caxis([0,0.66]); % 0,0.68<--60-->0.60
i
pause;
end

%%% >>>->>>->>> Off-bar Only >>>->>>->>>
h = fspecial('gaussian',4,3);
for i = 1:5:110
imgshow = reshape(VEavgbin_ra(i,:),3*4,5*4);
imgshow = conv2(imgshow,h,'same');
figure(1);
imagesc(imgshow');
axis equal;axis off;colormap('jet');caxis([0,0.76]); % 0,0.68<--60-->0.60
i
pause;
end
h = fspecial('gaussian',4,3);
for i = 96:5:120
imgshow = reshape(VEavgbin_ra(i,:),3*4,5*4);
imgshow = conv2(imgshow,h,'same');
figure(1);
imagesc(imgshow');
axis equal;axis off;colormap('jet');caxis([0,0.68]); % 0,0.68<--60-->0.60
i
pause;
end
