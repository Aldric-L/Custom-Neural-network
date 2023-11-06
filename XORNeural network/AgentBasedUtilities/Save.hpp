//
//  Save.hpp
//  AKML Project
//
//  Created by Aldric Labarthe on 05/11/2023.
//

#ifndef Save_hpp
#define Save_hpp

#include <iostream>
#include <string>

namespace akml {

    class Save {
        public:
        virtual std::string printAsCSV() = 0;
        static std::string printTitleAsCSV();
    };

}

#endif /* Save_hpp */
