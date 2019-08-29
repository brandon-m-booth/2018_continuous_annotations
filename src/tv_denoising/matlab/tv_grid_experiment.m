function tv_grid_experiment()
   lambdas = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10];
   for expon = [-4, -3, -2, -1, 0]
       for val = [2,3,4,5,6,7,8,9]
        lambdas = [lambdas, val*10^expon];
       end
   end
   eps = 1.0;
   % TODO - fix hard-coded paths
   tv_file_path = '/USC/2018_Continuous_Annotations/data/GreenIntensityTasks/tv/';
   gt_file_path = '/USC/2016_Continuous_Annotations/annotation_tasks/TaskA/AnnotationData/ground_truth_baselines/eval_dep/eval_dep_ground_truth_1hz.csv';
   gt_data = read_csv_file(gt_file_path, ',');
   header = gt_data(1,:);
   times = str2num(char(gt_data(2:end,1)));
   gt_data = str2num(char(gt_data(2:end,2)));
   for lambda=lambdas
       disp_str = sprintf('Working on lambda=%f', lambda);
       disp(disp_str);
       tv_sig = tv_1d(gt_data, lambda, eps);
       time_tv_mat = [times,tv_sig];
       tv_file = strcat(tv_file_path, sprintf('taskB_tv_%3.10f_lambda_1hz.csv', lambda))
       write_csv_file(tv_file, time_tv_mat, header);
       plot(gt_data, 'b--'); hold on; plot(tv_sig, 'r-');
   end
end
