clear all;
database=imageSet('k','recursive');
%r=imresize(read(img(1),1),[500 400]);
%c=ipcam('http://10.127.225.58:8080/videofeed');
c=webcam();
k=preview(c);
count=1;
s=zeros(5,300000);
face=vision.CascadeObjectDetector('FrontalFaceCART');
for i=1:10
frame=(snapshot(c));
bbox= (step(face,frame));
detected = insertShape(frame,'rectangle',bbox,'LineWidth',4);
if size(bbox,1)==1
cr=imcrop(frame,[bbox]);
k=imshow(rgb2gray(cr));
% imshow(detected);
end
s(i,1:size(extractHOGFeatures(cr),2))=extractHOGFeatures(cr);    
label{i}='test';
%count=count+1;
end
clear c;
for i=1:size(database,2)
    rsize(1,i)=size(extractHOGFeatures(imresize(read(database(i),1),[600 500])),2);
end
ms=max(rsize);
s1=zeros(size(database,2),ms);

for j=1:size(database,2)
    bbox1 = (step(face,read(database(j),1)));
    cr1=imcrop(read(database(j),1),bbox1);
s1(j,1:size(extractHOGFeatures(cr1),2))=extractHOGFeatures(cr1);    
label1{j}=database(j).Description;
%count=count+1;
end
faceClassifier = fitcecoc(s1,label1);
personLabel = predict(faceClassifier,s);
compare=strcmp(label,personLabel)
ind=find(compare);
figure;
    %subplot(2,1,1);imshow(query);title('Query Face');
    subplot(2,1,2);imshow(read(database(ind),1));title('Matched Class');
%%
faceClassifier = fitcecoc(s,label);
%%
%for j=1:size(img,2)
 %   rsize(1,j)=size(extractHOGFeatures(imresize(read(img(j),1),[400 500])),2);
%end

%ms=max(rsize);
s1=zeros(size(database,2),170000);

for j=1:size(database,2)
s1(j,1:size(extractHOGFeatures(imresize(read(database(j),1),[400 500])),2))=extractHOGFeatures(imresize(read(database(j),1),[400 500]));    
label1{j}=database(j).Description;
%count=count+1;
end
personLabel = predict(faceClassifier,s1);
%release(detected);
%%


parfor i=1:size(bbox,1)
    p(i)=size(extractHOGFeatures(imcrop(detected,bbox(i,:))),2);
 
end
j=zeros(size(bbox,1),max(p));

%p=zeros(3,80,80,3);
for i=1:size(bbox,1)
    j(i,1:size(extractHOGFeatures(imcrop(detected,bbox(i,:))),2))=extractHOGFeatures(imcrop(detected,bbox(i,:)));
%p(i,1:size(imcrop(detected,bbox(i,:)),1),1:size(imcrop(detected,bbox(i,:)),2),3)=(imcrop(detected,bbox(i,:)));

end
for i=1:size(bbox,1)
subplot(1,3,i);
imshow(imcrop(detected,bbox(i,:)));
%title('k','j','m');
end

