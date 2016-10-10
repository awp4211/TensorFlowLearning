#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <fstream>  
#include <string>  
#include <iostream>


#include "shlwapi.h"
#pragma comment(lib,"shlwapi.lib")


using namespace std;
using namespace cv;

String vidoes = "ufc11datasetfile.txt";





string resources_path = "D://PostGraduate0//YawnDetection//";

string train_output_path = "D://UFC11//train//";
string test_output_path = "D://UFC11//test//";

string image_format = ".jpg";
int index = 0;
int fps_index = 0;

const int output_width = 256, output_height = 256;

void saveimg(string dir, int fps_index,Mat frame);

int main(int argc, const char** argv)
{
	VideoCapture capture;
	Mat frame;
	ifstream test_file("ufc11_test_file.txt");
	ifstream train_file("ufc11_train_file.txt");

	string line;
	int current_type;
	string types[] ={ "basketball", "biking", "diving", 
		"golf_swing", "horse_riding", "soccer_juggling",
		"tennis_swing" ,"swing","trampoline_jumping",
		"volleyball_spiking", "walking" };

	int last_type = 0;

	if (test_file) // 有该文件  
	{
		CreateDirectory(test_output_path.c_str(), NULL);
		for (int i = 0; i < 11; i++)
		{
			string type_dir = test_output_path + types[i];
			CreateDirectory(type_dir.c_str(), NULL);
		}

		while (getline(test_file, line)) // line中不包括每行的换行符  
		{
			line = line.substr(0, line.find_first_of(' '));
			fps_index = 0;
			index++;

			//查看类别是否改变
			for (int i = 0; i < 11; i++)
			{
				if (line.find(types[i]) != -1)
				{
					current_type = i;
					break;
				}
			}
			if (current_type != last_type) index = 0;
			last_type = current_type;

			char p[32] = { 0 };
			sprintf(p, "//%d", index);

			string file2dir = test_output_path + types[current_type] + p;
			CreateDirectory(file2dir.c_str(), NULL);

			capture.open(line);

			cout << "*******************************************" << endl;
			cout << "** open file :" << line << endl;
			cout << "** file 2 dir:" << file2dir << endl;
			cout << "*****  start to process *******************" << endl;
			while (true)
			{
				capture.read(frame);
				if (!frame.empty())
				{
					saveimg(file2dir, fps_index, frame);
					fps_index++;
				}
				else
				{
					printf(" --(!) No captured frame -- Break!");
					break;
				}
				int c = waitKey(10);
				if ((char)c == 'c')
				{
					break;
				}
			}
		}
	}

	last_type = 0;
	index = 0;

	if (train_file) // 有该文件  
	{
		CreateDirectory(train_output_path.c_str(), NULL);
		for (int i = 0; i < 11; i++)
		{
			string type_dir = train_output_path + types[i];
			CreateDirectory(type_dir.c_str(), NULL);
		}

		while (getline(train_file, line)) // line中不包括每行的换行符  
		{
			line = line.substr(0, line.find_first_of(' '));
			fps_index = 0;
			index++;

			//查看类别是否改变
			for (int i = 0; i < 11; i++)
			{
				if (line.find(types[i]) != -1)
				{
					current_type = i;
					break;
				}
			}
			if (current_type != last_type) index = 0;
			last_type = current_type;

			char p[32] = { 0 };
			sprintf(p, "//%d", index);

			string file2dir = train_output_path + types[current_type] + p;
			CreateDirectory(file2dir.c_str(), NULL);

			capture.open(line);

			cout << "*******************************************" << endl;
			cout << "** open file :" << line << endl;
			cout << "** file 2 dir:" << file2dir << endl;
			cout << "*****  start to process *******************" << endl;
			while (true)
			{
				capture.read(frame);
				if (!frame.empty())
				{
					saveimg(file2dir, fps_index, frame);
					fps_index++;
				}
				else
				{
					printf(" --(!) No captured frame -- Break!");
					break;
				}
				int c = waitKey(10);
				if ((char)c == 'c')
				{
					break;
				}
			}
		}
	}


	

	return 0;
}

void saveimg(string dir,int fps_index, Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	resize(frame_gray, frame_gray, cvSize(output_height,output_width));

	// 1.创建文件夹

	// 2.创建文件名
	char p[32] = { 0 };
	sprintf(p, "//%d", fps_index);
	string str = p;
	// 3.输出图像
	string file_name = dir + str + image_format;
	//cout << "file_name = " << file_name << endl;
	imwrite(file_name, frame_gray);
}