#include "handle.h"

u_char* handle_IP
        (u_char *args,const struct pcap_pkthdr* pkthdr,const u_char*
        packet)
{
    const struct my_ip* ip;
    const struct my_tcp* tcp;
    u_int length = pkthdr->len;
    u_int hlen,version;
    u_int id, offset, proto, checksum;
//    int i;

    int len;

    /* jump pass the ethernet header */
    ip = (struct my_ip*)(packet + sizeof(struct ether_header));
    length -= sizeof(struct ether_header);

    /* check to see we have a packet of valid length */
    if (length < sizeof(struct my_ip))
    {
        printf("truncated ip %d",length);
        return NULL;
    }

    len      = ntohs(ip->ip_len);
    hlen     = IP_HL(ip); /* header length */
    version  = IP_V(ip);/* ip version */
    id		 = ntohs(ip->ip_id);
//    ttl		 = ntohs(ip->ip_ttl);
    proto	 = ntohs(ip->ip_p);
    checksum = ntohs(ip->ip_sum);

    /* check version */
    if(version != 4)
    {
        fprintf(stdout,"Unknown version %d\n",version);
        return NULL;
    }

    /* check header length */
    if(hlen < 5 )
    {
        fprintf(stdout,"bad-hlen %d \n",hlen);
    }

    /* see if we have as much packet as we should */
    if(length < len)
        printf("\ntruncated IP - %d bytes missing\n",len - length);

    /* Check to see if we have the first fragment */
    offset = ntohs(ip->ip_off);
    if((offset & 0x1fff) == 0 )/* aka no 1's in first 13 bits */
    {/* print SOURCE DESTINATION hlen version len offset */
        fprintf(stdout,"Internet Protocol: ");
        fprintf(stdout,"Src: %s, ",
                inet_ntoa(ip->ip_src));
        fprintf(stdout," Dst: %s\n",
                inet_ntoa(ip->ip_dst));
        fprintf(stdout,"     Version: %d\n", version);
        fprintf(stdout,"     Header Length: %d bytes\n", hlen*4);
        fprintf(stdout,"     Total Length: %d\n", len);
        fprintf(stdout,"     Identification: 0x%x (%d)\n", id,id);
        fprintf(stdout,"     TTL: %d\n", ip->ip_ttl);
        fprintf(stdout,"     Protocol: 0x%x (%d) %s\n", proto, proto, handle_protocol(ip->ip_p));
        fprintf(stdout,"     Checksum: 0x%x\n", checksum);

        tcp = (struct my_tcp*)(packet + sizeof(struct ether_header) + sizeof(struct my_ip));
        fprintf(stdout,"Transmission Control Protocol\n");
        fprintf(stdout,"     Src Port: %d, Dst Port: %d\n",  tcp->th_sport, tcp->th_dport);
        fprintf(stdout,"     Sequence Number: %u\n", ntohs(tcp->th_seq));
        fprintf(stdout,"     Windows Size: %d\n", ntohs(tcp->th_win));
        fprintf(stdout,"     Checksum: 0x%x\n", ntohs(tcp->th_sum));


    }
    return NULL;
}

/* handle ethernet packets, much of this code gleaned from
 * print-ether.c from tcpdump source
 */
u_int16_t handle_ethernet
        (u_char *args,const struct pcap_pkthdr* pkthdr,const u_char*
        packet)
{
    u_int caplen = pkthdr->caplen;
    struct ether_header *eptr;  /* net/ethernet.h */
    u_short ether_type;

    if (caplen < ETHER_HDRLEN)
    {
        fprintf(stdout,"Packet length less than ethernet header length\n");
        return -1;
    }

    /* lets start with the ether header... */
    eptr = (struct ether_header *) packet;
    ether_type = ntohs(eptr->ether_type);

    /* check to see if we have an ip packet */
    fprintf(stdout,"Captured length: %d bytes\n",caplen);
    if (ether_type == ETHERTYPE_IP)
    {
        fprintf(stdout,"Ethernet II \n");
        fprintf(stdout,"     Destination: %s\n"
                ,ether_ntoa((struct ether_addr*)eptr->ether_dhost));
        fprintf(stdout,"     Source: %s\n"
                ,ether_ntoa((struct ether_addr*)eptr->ether_shost));
        fprintf(stdout,"     Type: IP (0x0800)\n");
    }else{
        fprintf(stdout,"Not implemented type\n");
    }

    return ether_type;
}

char* handle_protocol
        (u_int8_t protocol)
{
    char *proto;
    switch(protocol)
    {
    case 1:
        sprintf(proto, "%s", "ICMP");
        break;
    case 2:
        sprintf(proto, "%s", "IGMP");
        break;
    case 6:
        sprintf(proto, "%s", "TCP");
        break;
    case 17:
        sprintf(proto, "%s", "UDP");
        break;
    }
    return proto;
}
