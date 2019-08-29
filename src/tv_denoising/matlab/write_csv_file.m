function write_csv_file(file_path, data_mat, header_array)
    T = array2table(data_mat);
    T.Properties.VariableNames = header_array;
    writetable(T, file_path, 'Delimiter', ',');
end