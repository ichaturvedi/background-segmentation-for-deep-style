dataSetDir = fullfile('.');
testImagesDir = fullfile(dataSetDir,'facesegtri/testpredrcnn/face200');
testLabelsDir = fullfile(dataSetDir,'facesegtri/testmasks');
imds = imageDatastore(testImagesDir);
classNames = ["face","background"];
labelIDs   = [255 0];
pxdsTruth = pixelLabelDatastore(testLabelsDir,classNames,labelIDs);
net = load('facenet');
net = net.net;
pxdsResults = semanticseg(imds,net,"WriteLocation",tempdir);
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTruth);
metrics
metrics.ClassMetrics
metrics.ConfusionMatrix

save facemetricsrcnn2.mat metrics