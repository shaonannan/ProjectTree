load('pythondata.mat')
lgn = 0.90 * reshape(gl',1,42*42*600);
save('gl.txt','lgn','-ASCII');
theta = reshape(theta,1,42*42);
phase = reshape(phase,1,42*42);
maxt = max(theta);mint = min(theta);
intt = (maxt-mint)/4.0;
clustorien = floor((theta-mint)/intt);
idx = find(clustorien==4);
clustorien(idx) = 3;
clusthyp = zeros(size(phase));
save('theta.txt','theta','-ASCII');
save('phase.txt','phase','-ASCII');
save('clustorien.txt','clustorien','-ASCII');
save('clusthyp.txt','clusthyp','-ASCII');

% vmem = reshape(lgn,41472,200);
% v_rec = zeros(size(vmem));
% % glgn = reshape(glgn',12800,250);
% % for i = 10:2:38
% h  = fspecial('gaussian',36,24);
% for i = 5:5:250
%     figure(334);%subplot(3,5,(i-1)/2-3);
%     img2show = conv2(reshape(vmem(:,i),24*12,24*6),h,'same');
%     imagesc(img2show(:,:));axis off;axis equal;colormap(jet);caxis([0.0 0.65]);
% %     img2show = conv2(reshape(vslave(:,i),160,80),h,'same');
% %     imagesc(img2show(:,:));axis off;axis equal;caxis([0.0 1.9]);  
%     pause;
% end

% spon  = 0.5*ones(12*20,150);
% ton_onset  = 0.025;
% toff_onset = 0.025;
% xbin = 1;
% tbin = 0.85;
% for i= 25:1:180
%     idt = 0.85*i;
%     idt = round(idt);
%     if idt>150
%         break;
%     end
% %     spoff[75+i-20,2*20-5:4*20+5]  = toff_onset + i* tbin * dt
%     spon((45+i-20):(75+i-20),idt)   = 1.0;
% end
% figure(1);imagesc(spon);axis equal;caxis([0,1.0]);colormap(gray);axis off;
