#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <algorithm>

#include <../include/base.h>

cell<uint32_t> * cellKeyList;
std::vector<cell<uint32_t>> keyList;

//Erzeuge verschiedene Werte für die Schlüssel und deren Längen zufällig
template <typename T1, typename T2>
std::vector<cell<T1,T2>> createCells(size_t cells_size, bool key_same=false){
    std::vector<cell<T1,T2>> cells_vector;
    cells_vector.reserve(cells_size);

    std::random_device generator;
    size_t seed = generator();
    std::mt19937 rnd(seed);

    if (key_same == false){
        T1 key = (T1) 1;
        T2 value = (T2) 1; 
        
        for (size_t i = 0; i < cells_size; i++){
            cells_vector.push_back(cell<T1,T2>{key, value});
            ++key;
            ++value;
        }
        std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
        return cells_vector;

    }else{
        std::uniform_int_distribution<T1> dist(1,cells_size);

        T1 key = dist(rnd); 
        T2 value = (T2) 1;
        
        for (size_t i = 0; i < cells_size; i++){
            cells_vector.push_back(cell<T1,T2>{key,value});
            ++value;
        }
        std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
        return cells_vector;
    }
};

//Lese eine Liste von Schlüsseln und deren Längen von einer Datei
template <typename T1, typename T2>
void readFile(char * file_name, size_t key_num){
    std::ifstream readfile(file_name,  std::ios::out |  std::ios::binary);
    
    if(!readfile) {
        std::cout << "Die Datei kann leider nicht geöffnet werden." <<  std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < key_num; i++){
        cell<T1,T2> key;
        readfile.read((char*) &key, sizeof(cell<T1,T2>));
        std::cout << key.key << ","  << key.value << std::endl;
    }
    
    
    readfile.close();
    
    if(!readfile.good()) {
        std::cout << "Fehler beim Lesen von Schlüsseln" << std::endl;
        exit(EXIT_FAILURE);
    }
};

//Schreibe eine Liste von Schlüsseln und deren Längen auf einer Datei
template <typename T1, typename T2>
void writeFile(char * file_name, size_t key_num, bool key_same){
    std::ofstream writefile(file_name, std::ios::out | std::ios::binary);
    
    if (!writefile){
        std::cout << "Die Datei kann leider nicht geöffnet werden." << std::endl;
        exit (EXIT_FAILURE);
    }

    keyList = createCells<T1,T2>(key_num,key_same);
    cellKeyList = keyList.data();
    
    for (size_t i = 0; i < key_num; i++)
        writefile.write((char *) &cellKeyList[i], sizeof(cell<T1,T2>));

    writefile.close();
    
    if(!writefile.good()) {
        std::cout << "Fehler bei der Eintragung von Schlüsseln" << std::endl;
        exit (EXIT_FAILURE);
    }
};

int main(int argc, char** argv){
    size_t key_num;
    char * file_name;
    int int_key_same, int_read_write;
    bool key_same, read_write;
    std::string file_name_str;
    
    if(argc < 5){
        std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
        return -1;
    }

    int_read_write = atoi(argv[1]);
    key_num = (size_t) atoi(argv[2]);
    file_name = argv[3];
    int_key_same = atoi(argv[4]);

    if (int_read_write<0 || int_read_write>1){
        std::cout << "Der Code des Schreibens oder Lesens von Datei muss entweder 0 bis 1 sein." << std::endl;
        return -1;
    }

    if (key_num <=0){
        std::cout << "Die Anzahl an Schlüssel muss mehr als Null betragen." << std::endl;
        return -1;
    }

    file_name_str.append(file_name);

    if (file_name_str.size()==0){
        std::cout << "Es muss einen Namen für eine Datei geben." << std::endl;
        return -1;
    }

    if (int_key_same<0 || int_key_same>1){
        std::cout << "Der Code der Gleichheit der Schlüsselgröße muss entweder 0 bis 1 sein." << std::endl;
        return -1;
    }
    
    if (int_read_write == 1){
        read_write = true;
    }else{
        read_write = false;     
    }

    if (int_key_same == 1){
        key_same = true;
    }else{
        key_same = false;     
    }

    if (read_write == true){
        writeFile<uint32_t,uint32_t>(file_name, key_num, key_same);
    }else{
        readFile<uint32_t,uint32_t>(file_name, key_num);
    }

    return 0;
};