    unordered_map<int, int> mymap;
        mymap.insert({2,4});
            mymap.insert({3,6});
                unordered_map<int, int>::const_iterator got;
                    got = mymap.find(2);
                        if (got == mymap.end()) {
                                    cout << "Not find " << endl;
                                        }else
    {
                cout << got->second << endl;
                    }
    got = mymap.find(1);
        if (got == mymap.end()) {
                    cout << "Not find " << endl;
                        }else
    {
                cout << got->second << endl;
                    }

