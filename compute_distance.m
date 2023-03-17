%% compute distance matrix from the x-y-z coordinates in ALLEEG
clc;clear;
%% load channel locations
load('64channel_location.mat');
%% from ALLEEG format to matrix format
coordinates = zeros(64, 3);
for i = 1:64
    coordinates(i,1) = channelloc(i).X;
    coordinates(i,2) = channelloc(i).Y;
    coordinates(i,3) = channelloc(i).Z;
end
%% compute distance matrix from coordinates
dist_m = pdist(coordinates);
dist_m = squareform(dist_m);
writematrix(dist_m,'64chans_distmat.csv');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DIFF SYSTEM
clc;clear;
%% please Select-process different part of this script
%% load channel locations
load('32channel_location.mat');
%% from ALLEEG format to matrix format
coordinates = zeros(32, 3);
for i = 1:32
    coordinates(i,1) = location(i).X;
    coordinates(i,2) = location(i).Y;
    coordinates(i,3) = location(i).Z;
end
%% compute distance matrix from coordinates
dist_m = pdist(coordinates);
dist_m = squareform(dist_m);
writematrix(dist_m,'32chans_distmat.csv');
