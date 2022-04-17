#include <iostream>
#include <map>
#include <string>
#include <cstring>
#include<math.h>
using namespace std;
const int N = 256;
int len; //串长度
int cnt; //字符数
string str;
int char_num[N];//统计字符
struct node {//字符及概率区间
	char c;
	double l, r;  //区间上下限
}ch[N];
struct mydata {
	double low=0.0, high=1.0, delta;  //概率区间上下限及长度
	mydata() = default;
	mydata(double a, double b) :low(a), high(b), delta(b-a) {}
};
map<char, mydata> mp;

void init() {
	memset(char_num, 0, sizeof(char_num));
	printf("输入使用的字符总数n\n");
	cin >> cnt;
	double last = 0.0;  // 字典序上一个字母频率的右边界
	string s;
	printf("输入字符\n");
	cin >> s;
	for (int i = 0; i < cnt; i ++){
		char c = s[i];
		double p;
		printf("输入字符 %c 的概率\n", c);
		cin >> p;
		mp.insert(make_pair(c, mydata(last, last+p)));
		last += p;
	}
	printf("输入串总字符数n\n");
	cin >> len;
	printf("输入字符串：\n");
	cin >> str;

}

string encode(double &db,string str){
	double low = 0.0;
	double high = 1.0;
	// 左右区间迭代
	for (auto it = str.begin(); it != str.end(); it++) {
		double delta = high - low;
		high = low + delta * mp[*it].high;
		low = low + delta * mp[*it].low;
	}
	//寻找最短二进制：不断增加2^i，第一次落在区间内则为最短
	string anstr = "";
	double ans = 0.0;
	int cnt = 1;
	while (ans < low) {
		ans += pow(0.5, cnt);
		anstr += '1';
		if (ans >= high) {
			ans -= pow(0.5, cnt);
			anstr[cnt-1] = '0';
		}
		cnt++;
	}
	db = ans;
	return anstr;
}

string decode(double value) {
	double low, high;                   //临时译码区间
	double prelow = 0.0, prehigh = 1.0;  //前一个字符的译码区间
	string ans = "";
	int cur = 0;
	while (true) {
		low = prelow;
		high = prehigh;
		for (auto it = mp.begin(); it != mp.end(); it++) {
			// 根据编码值与概率不断解码
			double delta = high - low;
			high = low + delta * it->second.high;
			low = low + delta * it->second.low;
			if (value>=low && value<high) {
				prelow = low;
				prehigh = high;
				ans += (it->first);
				cur++;
				break;
			}else
			{
				low = prelow;
				high = prehigh;
			}
		}
		if (cur == len)break;
	}
	return ans;
}

int main() {
	mydata sp;
	init();
	cout << "\n";
	for (auto it = mp.begin(); it != mp.end(); it++) {
		cout << it->first <<  " (";
		cout << it->second.low << " ";
		cout << it->second.high << ") ";
		cout << it->second.delta << "\n";
	}
	
	double db;
	string anstr = encode(db, str);
	cout << "编码后小数表示:\n" << db << endl;
	cout << "\n";
	cout << "最短二进制编码:\n" << anstr << endl;
	cout << "\n";
	string destr=decode(db);
	cout <<"解码后的字符串: \n"<< destr << endl;

	return 0;
}