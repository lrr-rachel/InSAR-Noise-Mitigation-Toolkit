% input data path
Datapath = '../UNet_CV/results/model/';
% output grayscale image path
Graypath = '../UNet_CV/results/Model_wrapped/';

% output rgb image path
RGBpath = '../UNet_CV/results/Color_wrapped/';

if ~exist(Graypath, 'dir')
       mkdir(Graypath)
end
if ~exist(RGBpath, 'dir')
       mkdir(RGBpath)
end

towrap(Datapath,Graypath)
torgb(Graypath,RGBpath)

% convert unwrapped to wrapped
function grayI = towrap(Datapath,Savepath)
    filelist = dir(fullfile(Datapath,'*.png'));
    for i = 1 : 1 :length(filelist)
        I = imread([Datapath,filelist(i).name]);
        I = im2double(I);
        % convert grayscale to rad interferogram
        radI = I*100 - 50;
        % wrapping
        wrappedI = wrapTo2Pi(radI);
        % rescale
        grayI = wrappedI/(2*pi);
        imshow(grayI)
        imwrite(grayI,[Savepath,filelist(i).name],'png');
    end
end

% convert gray to rgb
function rgbI = torgb(Graypath,RGBpath)
    filelist = dir(fullfile(Graypath,'*.png'));
    for i = 1 : 1 :length(filelist)
        grayI = imread([Graypath,filelist(i).name]);
        rgbI = ind2rgb(grayI,jet(256));
        imshow(rgbI)
        imwrite(rgbI,[RGBpath,filelist(i).name],'png');
    end
end




