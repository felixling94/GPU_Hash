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
template <typename T>
std::vector<cell<T>> createCells(size_t cells_size, bool key_length_same=false){
    std::vector<cell<T>> cells_vector;
    cells_vector.reserve(cells_size);

    std::random_device generator;
    size_t seed = generator();
    std::mt19937 rnd(seed);

    if (key_length_same == false){
        T key = (T) 1;
        T key_length = (T) 1; 
        
        for (size_t i = 0; i < cells_size; i++){
            cells_vector.push_back(cell<T>{key,key_length});
            ++key;
            ++key_length;
        }
        std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
        return cells_vector;

    }else{
        std::uniform_int_distribution<T> dist(1,cells_size);

        T key = (T) 1;
        T key_length = dist(rnd); 
        
        for (size_t i = 0; i < cells_size; i++){
            cells_vector.push_back(cell<T>{key,key_length});
            ++key;
        }
        std::shuffle(cells_vector.begin(), cells_vector.end(), rnd);
        return cells_vector;
    }
};

//Lese eine Liste von Schlüsseln und deren Längen von einer Datei
template <typename T>
void readFile(char * file_name, size_t key_num){
    std::ifstream readfile(file_name,  std::ios::out |  std::ios::binary);
    
    if(!readfile) {
        std::cout << "Die Datei kann leider nicht geöffnet werden." <<  std::endl;
        exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < key_num; i++){
        cell<T> key;
        readfile.read((char*) &key, sizeof(cell<T>));
        std::cout << key.key << "  "  << key.key_length << std::endl;
    }
    
    
    readfile.close();
    
    if(!readfile.good()) {
        std::cout << "Fehler beim Lesen von Schlüsseln" << std::endl;
        exit(EXIT_FAILURE);
    }
};

//Schreibe eine Liste von Schlüsseln und deren Längen auf einer Datei
template <typename T>
void writeFile(char * file_name, size_t key_num, bool key_length_same){
    std::ofstream writefile(file_name, std::ios::out | std::ios::binary);
    
    if (!writefile){
        std::cout << "Die Datei kann leider nicht geöffnet werden." << std::endl;
        exit (EXIT_FAILURE);
    }

    keyList = createCells<T>(key_num,key_length_same);
    cellKeyList = keyList.data();
    
    for (size_t i = 0; i < key_num; i++)
        writefile.write((char *) &cellKeyList[i], sizeof(cell<T>));

    writefile.close();
    
    if(!writefile.good()) {
        std::cout << "Fehler bei der Eintragung von Schlüsseln" << std::endl;
        exit (EXIT_FAILURE);
    }
};

int main(int argc, char** argv){
    size_t key_num;
    char * file_name;
    int int_key_length_same, int_read_write;
    bool key_length_same, read_write;
    std::string file_name_str;
    
    if(argc < 5){
        std::cout << "Fehler bei der Eingabe von Parametern" << std::endl;
        return -1;
    }

    int_read_write = atoi(argv[1]);
    key_num = (size_t) atoi(argv[2]);
    file_name = argv[3];
    int_key_length_same = atoi(argv[4]);

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

    if (int_key_length_same<0 || int_key_length_same>1){
        std::cout << "Der Code der Gleichheit der Schlüsselgröße muss entweder 0 bis 1 sein." << std::endl;
        return -1;
    }
    
    if (int_read_write == 1){
        read_write = true;
    }else{
        read_write = false;     
    }

    if (int_key_length_same == 1){
        key_length_same = true;
    }else{
        key_length_same = false;     
    }

    if (read_write == true){
        writeFile<uint32_t>(file_name, key_num, key_length_same);
    }else{
        readFile<uint32_t>(file_name, key_num);
    }

    return 0;
};