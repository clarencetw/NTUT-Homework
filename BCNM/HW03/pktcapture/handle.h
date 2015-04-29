#ifndef HANDLE_H_
#define HANDLE_H_

#include <pcap.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <signal.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <netinet/if_ether.h>
#include <net/ethernet.h>
#ifdef __linux__
#include <netinet/ether.h>
#endif
#include <netinet/ip.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#ifndef ETHER_HDRLEN
#define ETHER_HDRLEN 14
#endif

u_int16_t handle_ethernet
		(u_char *args,const struct pcap_pkthdr* pkthdr,const u_char*
		packet);

u_char* handle_IP
		(u_char *args,const struct pcap_pkthdr* pkthdr,const u_char*
		packet);

char* handle_protocol
		(u_int8_t my_ip);

struct my_ip {
	u_int8_t	ip_vhl;		/* header length, version */
#define IP_V(ip)	(((ip)->ip_vhl & 0xf0) >> 4)
#define IP_HL(ip)	((ip)->ip_vhl & 0x0f)
	u_int8_t	ip_tos;		/* type of service */
	u_int16_t	ip_len;		/* total length */
	u_int16_t	ip_id;		/* identification */
	u_int16_t	ip_off;		/* fragment offset field */
#define	IP_DF 0x4000		/* dont fragment flag */
#define	IP_MF 0x2000		/* more fragments flag */
#define	IP_OFFMASK 0x1fff	/* mask for fragmenting bits */
	u_int8_t	ip_ttl;		/* time to live */
	u_int8_t	ip_p;		/* protocol */
	u_int16_t	ip_sum;		/* checksum */
	struct	in_addr ip_src,ip_dst;	/* source and dest address */
};

struct my_tcp {
	u_int16_t th_sport;		/* source port */
	u_int16_t th_dport;		/* destination port */
	u_int32_t th_seq;		/* sequence number */
	u_int32_t th_ack;		/* acknowledgement number */
	u_char th_offx2;		/* data offset, rsvd */
#define TH_OFF(th)	(((th)->th_offx2 & 0xf0) >> 4)
	u_char th_flags;
#define TH_FIN 0x01
#define TH_SYN 0x02
#define TH_RST 0x04
#define TH_PUSH 0x08
#define TH_ACK 0x10
#define TH_URG 0x20
#define TH_ECE 0x40
#define TH_CWR 0x80
#define TH_FLAGS (TH_FIN|TH_SYN|TH_RST|TH_ACK|TH_URG|TH_ECE|TH_CWR)
	u_int16_t th_win;		/* window */
	u_int16_t th_sum;		/* checksum */
	u_int16_t th_urp;		/* urgent pointer */
};


#endif /*HANDLE_H_*/
