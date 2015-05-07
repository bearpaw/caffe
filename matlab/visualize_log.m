function visualize_log(logfile)
if nargin < 1
  logfile = '/home/wyang/github/caffe/examples/lsp_window_data/conv5-96c/cache/train-11-Feb-2015-conv5-96/train-11-Feb-2015-conv5-96-train.log';
end

[~, filename, ~] = fileparts(logfile);

if ~exist(['cache/' filename], 'dir')
  mkdir(['cache/' filename]);
end

system(['../tools/extra/parse_log.sh ' logfile]);
movefile([filename '.log.train'], ['cache/' filename '/' filename '.train']);
movefile([filename '.log.test'], ['cache/' filename '/' filename '.test']);

nametrain = ['cache/' filename '/' filename '.train'];
nametest = ['cache/' filename '/' filename '.test'];
SumN = 50;

% parse TRAIN file
fid     = fopen(nametrain);
title   = textscan(fid, '%s', 5);
content = textscan(fid, '%f %f %f %f');
train   = cell2struct(content, {'iter', 'time', 'loss', 'lr'}, 2);
fclose(fid);

% parse TEST file
fid     = fopen(nametest);
title   = textscan(fid, '%s', 5);
content = textscan(fid, '%f %f %f %f');
test   = cell2struct(content, {'iter', 'time', 'accuracy', 'loss'}, 2);
fclose(fid);

% compute SMOOTH train loss
train_err_smooth = train.loss;
for j = 1:length(train.loss)
    startj = max(1, j - SumN);
    endj = min(length(train.loss), j+SumN);
    train_err_smooth(j) = mean(train.loss(startj:endj));
end


% compute SMOOTH test loss
test_err_smooth = test.loss;
for j = 1:length(test.loss)
    startj = max(1, j - 5);
    endj = min(length(test.loss), j+5);
    test_err_smooth(j) = mean(test.loss(startj:endj));
end
% draw 
f = figure('name', 'LOSS');  hold on;

plot(train.iter(3:end), train.loss(3:end), 'g');     % training loss
plot(train.iter(3:end), train_err_smooth(3:end), 'b', 'LineWidth', 2); % smooth training loss

plot(test.iter(1:end), test.loss(1:end), '-xr', 'LineWidth', 2); % testing loss
% plot(test.iter(3:end), test_err_smooth(3:end), 'k', 'LineWidth', 2); % smooth testing loss
plot(test.iter(1:end), test.accuracy(1:end), '-xk', 'LineWidth', 2); % testing accuracy
legend('Train Loss', 'Train Loss', 'Test Loss', 'Test Accuracy');

hold off;
saveas(gcf, ['cache/' filename '/' filename '.png']);

system(['rm -rf ' nametrain ' ' nametest]);

[mintestloss, idx] = min(test.loss);
[maxtestacc, accidx] = max(test.accuracy);
fprintf('Min Test Loss is %.6f at # %6d iteration.\n', mintestloss, test.iter(idx));
fprintf('Max Test Accuracy is %.6f at # %6d iteration.\n', maxtestacc, test.iter(accidx));
pause; close;
