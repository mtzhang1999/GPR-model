function [processed_data, data_ave] = Data_Preprocess(file_name, len, inverse)
    data = csvread(file_name, 2, 1);
    processed_data = flipud(data(:, 1));
    if len ~= 0
        if inverse
            processed_data = processed_data(end-len+1:end);
        else
            processed_data = processed_data(1:len);
        end
    end
    data_ave = mean(processed_data);
end